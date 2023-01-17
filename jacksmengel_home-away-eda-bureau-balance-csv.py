# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

# Dataframe legend
# bureau_application = train data
# bureau = all loans (+ meta data)
# closed_loans = closed loans
# active_loans = active loans
# bureau_balance = loan payment history
# u = loans w/ no payment history
# uah_out = list of active loans w/ payment history
# uch_out = list of closed loans w/ payment history
# unique_loans_sample = active/closed loans evaluated
# bureau_with_scores = active/closed loans evaluated with meta data
# unique_SK_ID_CURR = ML input (steps numbered)
# bureau_application_scores = adding metrics to training data

# Function legend
# dpd = function to find highest streak of missing payments
# For loop 1: Score closed loans (good boys)
# For loop 2: Score active loans (potentially bad boys)
# Train model

# Description of model
# TBD


# bureau_application = train data
# b_a_u_out = unique SK_ID_CURR
# 122 columns, 307,511 rows
bureau_application = pd.read_csv('../input/application_train.csv')
b_a_u = bureau_application.SK_ID_CURR.unique()

# bureau = all loans (+ meta data)
bureau_staging = pd.read_csv('../input/bureau.csv')

bureau = bureau_staging[bureau_staging.SK_ID_CURR.isin(b_a_u)]

# closed_loans = closed loans
closed_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']

# active_loans = active loans
active_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Active']

# add on a column to store each loan's score
active_loans['LOAN_SCORE'] = -99
closed_loans['LOAN_SCORE'] = -99

# bureau_balance = loan payment history
bureau_balance = pd.read_csv('../input/bureau_balance.csv')

# u = loans w/ no payment history
unique_loans_history = bureau_balance.SK_ID_BUREAU.unique()
s = {'SK_ID_BUREAU_S': unique_loans_history}
t = pd.DataFrame(data = s)

unique_bureau = bureau.filter(['SK_ID_BUREAU'], axis = 1)

r = pd.merge(
    unique_bureau, # all loans listed
    t, # All loans which have payment history
    left_on = 'SK_ID_BUREAU',
    right_on = 'SK_ID_BUREAU_S',
    how = 'left'
)

u = r[r['SK_ID_BUREAU_S'].isnull()]

# uah_out = active loans w/ payment history
active_bureau_join = pd.merge(
    active_loans,
    bureau_balance,
    on = 'SK_ID_BUREAU',
    how = 'inner'
)

unique_active_history = active_bureau_join.SK_ID_BUREAU.unique()
uah = {'SK_ID_BUREAU': unique_active_history}
uah_out = pd.DataFrame(data = uah)

# uch_out = closed loans w/ payment history
closed_bureau_join = pd.merge(
    closed_loans,
    bureau_balance,
    on = 'SK_ID_BUREAU',
    how = 'inner'
)

unique_closed_history = closed_bureau_join.SK_ID_BUREAU.unique()
uch = {'SK_ID_BUREAU': unique_closed_history}
uch_out = pd.DataFrame(data = uch)

# How many SK_ID_CURR in application_train don't have loan history?
# b_out = DISTINCT applicants in bureau_application
b = bureau_application.SK_ID_CURR.unique()
b_all = {'SK_ID_CURR_ALL': b}
b_out = pd.DataFrame(data = b_all)

# usicwh_out = list of applicants w/ loan history
c = uch_out.append(uah_out, ignore_index=True)
c_join_b = pd.merge(
    c,
    bureau,
    on = 'SK_ID_BUREAU',
    how = 'inner'
)
unique_sk_id_curr_w_history = c_join_b.SK_ID_CURR.unique()
usicwh = {'SK_ID_CURR_W_PAYMENT_HIST': unique_sk_id_curr_w_history}
usicwh_out = pd.DataFrame(data = usicwh)

# b_miss = list of applicants w/ no payment history
u_join_b = pd.merge(
    bureau_application,
    usicwh_out,
    left_on = 'SK_ID_CURR',
    right_on = 'SK_ID_CURR_W_PAYMENT_HIST',
    how = 'left'
)
b_miss = u_join_b[u_join_b['SK_ID_CURR_W_PAYMENT_HIST'].isnull()]

# print outcome
print('# of rows in bureau_application: ', bureau_application.shape)
print('# of applications: ', b_out.shape)
print('# of applications with loan payment history: ', usicwh_out.shape)
print('# of applications with NO loan payment history: ', b_miss.shape)

# dpd = function to find highest streak of missing payments
high_dpd = 0

def dpd(row):
    global high_dpd
    
    if row['STATUS'] == '1':
        if high_dpd < 1:
            high_dpd = 1
    elif row['STATUS'] == '2':
        if high_dpd < 2:
            high_dpd = 2
    elif row['STATUS'] == '3':
        if high_dpd < 3:
            high_dpd = 3
    elif row['STATUS'] == '4':
        if high_dpd < 4:
            high_dpd = 4
    elif row['STATUS'] == '5':
        if high_dpd < 5:
            high_dpd = 5

# Score closed loans (good boys)
count_outer = 0
for index, row in uch_out.iterrows():
    
    df = closed_bureau_join[closed_bureau_join['SK_ID_BUREAU'] == row['SK_ID_BUREAU']]
    df = df.sort_values(['MONTHS_BALANCE'], ascending = True)
    
    score = .5
    high_dpd = 0
    
    df.apply(dpd, axis = 1)
    
    if high_dpd == 0:
        score = 1
    if high_dpd == 1:
        score = .6
    if high_dpd == 2:
        score = .4
    if high_dpd == 3:
        score = .2
    if high_dpd == 4:
        score = .1
    if high_dpd == 5:
        score = .05
        
    closed_loans.loc[closed_loans['SK_ID_BUREAU'] == row['SK_ID_BUREAU'], 'LOAN_SCORE'] = score
    
    # for performance purposes, only want subset of loans to test theory
    count_outer = count_outer + 1
    print(count_outer)
    if count_outer == 1000:
        break

# Score active loans (potentially bad boys)
foreclosed_indicator = 0
def foreclosed(row):
    global foreclosed_indicator
    if row['STATUS'] == 'C':
        if row['AMT_CREDIT_SUM_DEBT'] > 0:
            foreclosed_indicator = 1

count_outer_active = 0
for index, row in uah_out.iterrows():
    
    df = active_bureau_join[active_bureau_join['SK_ID_BUREAU'] == row['SK_ID_BUREAU']]
    df = df.sort_values(['MONTHS_BALANCE'], ascending = True)
    
    score = .5
    high_dpd = 0
    foreclosed_indicator = 0
    
    df.apply(dpd, axis = 1)
    df.apply(foreclosed, axis = 1)

    if foreclosed_indicator == 0:
        if high_dpd == 0:
            score = .95
        if high_dpd == 1:
            score = .5
        if high_dpd == 2:
            score = .4
        if high_dpd == 3:
            score = .3
        if high_dpd == 4:
            score = .2
        if high_dpd == 5:
            score = .1
    elif foreclosed_indicator == 1:
        score = 0
        
    active_loans.loc[active_loans['SK_ID_BUREAU'] == row['SK_ID_BUREAU'], 'LOAN_SCORE'] = score
    
    # for performance purposes, only want subset of loans to test theory
    count_outer_active = count_outer_active + 1
    print(count_outer_active)
    if count_outer_active == 1000:
        break

# unique_loans_sample = active/closed loans evaluted
combine = closed_loans.append(active_loans, ignore_index=True)
unique_loans_sample = combine[combine['LOAN_SCORE'] != -99]

print('unique_loans_sample: ', unique_loans_sample.shape)

# bureau_with_scores = active/closed loans evaluated with meta data
# unique_SK_ID_CURR = ML input (steps numbered)
# 3. bureau_application_scores = adding metrics to training data
bureau_with_scores = pd.merge(
    bureau,
    unique_loans_sample,
    on = 'SK_ID_BUREAU',
    how = 'inner'
)

print('bureau_with_scores: ', bureau_with_scores.shape)

# unique_SK_ID_CURR = ML input (steps numbered)

# 1. list of all unique SK_ID_CURR
a2 = bureau_with_scores.SK_ID_CURR_x.unique()
d2 = {'SK_ID_CURR': a2}
unique_SK_ID_CURR = pd.DataFrame(data = d2)

print('unique_SK_ID_CURR pre metrics: ', unique_SK_ID_CURR.shape)

# 2. layer on metrics using bureau_with_scores:
# count
# high score
# low score
# foreclosure flag
# average score
# median score

# count
loan_count_linear = bureau_with_scores.groupby(['SK_ID_CURR_x'])['LOAN_SCORE'].count().reset_index(name = 'LOAN_COUNT')
unique_SK_ID_CURR = pd.merge(
    unique_SK_ID_CURR,
    loan_count_linear,
    left_on = 'SK_ID_CURR',
    right_on = 'SK_ID_CURR_x',
    how = 'inner'
)

# high score
max_loan_score_linear = bureau_with_scores.groupby(['SK_ID_CURR_x'])['LOAN_SCORE'].max().reset_index(name = 'MAX_LOAN_SCORE_LINEAR')
unique_SK_ID_CURR = pd.merge(
    unique_SK_ID_CURR,
    max_loan_score_linear,
    left_on = 'SK_ID_CURR',
    right_on = 'SK_ID_CURR_x',
    how = 'inner'
)

# low score 
min_loan_score_linear = bureau_with_scores.groupby(['SK_ID_CURR_x'])['LOAN_SCORE'].min().reset_index(name = 'MIN_LOAN_SCORE_LINEAR')

# foreclosure check
def foreclose_classification(row):
    if row['MIN_LOAN_SCORE_LINEAR'] > 0:
        return 0
    else:
        return 1
        
min_loan_score_linear['FORECLOSURE_IN_PAST'] = min_loan_score_linear.apply(foreclose_classification, axis = 1)

unique_SK_ID_CURR = pd.merge(
    unique_SK_ID_CURR,
    min_loan_score_linear,
    left_on = 'SK_ID_CURR',
    right_on = 'SK_ID_CURR_x',
    how = 'inner'
)

# average score
average_loan_score_linear = bureau_with_scores.groupby(['SK_ID_CURR_x'])['LOAN_SCORE'].mean().reset_index(name = 'AVERAGE_LOAN_SCORE_LINEAR')
unique_SK_ID_CURR = pd.merge(
    unique_SK_ID_CURR,
    average_loan_score_linear,
    left_on = 'SK_ID_CURR',
    right_on = 'SK_ID_CURR_x',
    how = 'inner'
)

# median score
median_loan_score_linear = bureau_with_scores.groupby(['SK_ID_CURR_x'])['LOAN_SCORE'].median().reset_index(name = 'MEDIAN_LOAN_SCORE_LINEAR')
unique_SK_ID_CURR = pd.merge(
    unique_SK_ID_CURR,
    median_loan_score_linear,
    left_on = 'SK_ID_CURR',
    right_on = 'SK_ID_CURR_x',
    how = 'inner'
)

print('unique_SK_ID_CURR post metrics: ', unique_SK_ID_CURR.shape)

# 3. bureau_application_scores = adding metrics to training data
bureau_application_scores = pd.merge(
    unique_SK_ID_CURR,
    bureau_application,
    on = 'SK_ID_CURR',
    how = 'inner'
)

print('bureau_application_scores: ', bureau_application_scores.shape)

bureau_application_scores.to_csv('2018-08-12_pre_processing.csv', index = False)

# 4. visualize
count = 'LOAN_COUNT'
avg_linear = 'AVERAGE_LOAN_SCORE_LINEAR'
min_linear = 'MIN_LOAN_SCORE_LINEAR'
max_linear = 'MAX_LOAN_SCORE_LINEAR'
median_linear = 'MEDIAN_LOAN_SCORE_LINEAR'
foreclosure = 'FORECLOSURE_IN_PAST'
Y = 'TARGET'

bad_boys = bureau_application_scores[bureau_application_scores[Y] == 1]
good_boys = bureau_application_scores[bureau_application_scores[Y] == 0]

x_axis_avg_linear = bureau_application_scores[avg_linear]
x_axis_min_linear = bureau_application_scores[min_linear]
x_axis_max_linear = bureau_application_scores[max_linear]
x_axis_median_linear = bureau_application_scores[median_linear]
x_axis_count = bureau_application_scores[count]
x_axis_foreclosure = bureau_application_scores[foreclosure]

y_axis = bureau_application_scores[Y]

#plt.bar(x_axis_avg_linear, y_axis)
#plt.xlabel("x_axis_avg_linear")

bad_boys.hist(column = foreclosure)
good_boys.hist(column = foreclosure)

bad_boys.hist(column = avg_linear)
good_boys.hist(column = avg_linear)

#plt.scatter(x_axis_min_linear, y_axis)
#plt.xlabel("x_axis_min_linear")

bad_boys.hist(column = min_linear)
good_boys.hist(column = min_linear)

#plt.scatter(x_axis_max_linear, y_axis)
#plt.xlabel("x_axis_max_linear")

bad_boys.hist(column = max_linear)
good_boys.hist(column = max_linear)

#plt.scatter(x_axis_median_linear, y_axis)
#plt.xlabel("x_axis_median_linear")

bad_boys.hist(column = median_linear)
good_boys.hist(column = median_linear)

#plt.scatter(x_axis_count, y_axis)
#plt.xlabel("x_axis_count")
#plt.show()

bad_boys.hist(column = count)
good_boys.hist(column = count)

# 5. Train model
x_cols = [
    'SK_ID_CURR',
    'LOAN_COUNT',
    'MAX_LOAN_SCORE_LINEAR',
    'MIN_LOAN_SCORE_LINEAR',
    'AVERAGE_LOAN_SCORE_LINEAR',
    'MEDIAN_LOAN_SCORE_LINEAR',
    'FORECLOSURE_IN_PAST'
]

y_cols = ['TARGET']

X = bureau_application_scores[x_cols]
Y = bureau_application_scores[y_cols]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .33, random_state = 0)

model = RandomForestClassifier()
model.fit(x_train, y_train.values.ravel())
predictions = model.predict_proba(x_test)

# validate
preds = predictions[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(
    fpr, 
    tpr, 
    color='darkorange',
    lw=lw, 
    label='ROC curve (area = %0.2f)' % roc_auc
)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# HOME CREDIT
# POS_CASH_BALANCE: Monthly balances of loans with Home Credit
# credit_card_balance: Monthly balances of credit cards with Home Credit
# previous_application: All previous applications for clients in sample
# installments_payments: Repayment history for previous credits.  One row for each payment OR missed payment


