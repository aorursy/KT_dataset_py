import numpy as np

import pandas as pd

pd.set_option('max_columns', 120)

pd.set_option('max_colwidth', 5000)

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = (12,8)
loan = pd.read_csv("../input/lending-club/loan.csv",encoding = "ISO-8859-1", low_memory=False)

loan.head(3)
loan.shape
loan.describe()
loan.head()
missing = round(100*(loan.isnull().sum()/len(loan.id)), 2)

missing.loc[missing > 0]
columns_with_missing_values = list(missing[missing >= 50].index)

len(columns_with_missing_values)
loan = loan.drop(columns_with_missing_values,axis=1)

loan.shape
missing = round(100*(loan.isnull().sum()/len(loan.id)), 2)

missing[missing != 0]
loan = loan.drop('desc',axis=1)
print("unique emp_title : %d"  % len(loan.emp_title.unique()))

print("unique emp_length : %d"  % len(loan.emp_length.unique()))

print("unique title : %d"  % len(loan.title.unique()))

print("unique revol_util : %d"  % len(loan.revol_util.unique()))

print("unique title : %d"  % len(loan.title.unique()))

print("unique last_pymnt_d : %d"  % len(loan.last_pymnt_d.unique()))

print("unique last_credit_pull_d : %d"  % len(loan.last_credit_pull_d.unique()))

print("unique collections_12_mths_ex_med : %d"  % len(loan.collections_12_mths_ex_med.unique()))

print("unique chargeoff_within_12_mths : %d"  % len(loan.chargeoff_within_12_mths.unique()))

print("unique pub_rec_bankruptcies : %d"  % len(loan.pub_rec_bankruptcies.unique()))

print("unique tax_liens : %d"  % len(loan.tax_liens.unique()))
loan.emp_length.unique()
loan.collections_12_mths_ex_med.unique()
loan.chargeoff_within_12_mths.unique()
loan.pub_rec_bankruptcies.unique()
loan.tax_liens.unique()
drop_columnlist = ['collections_12_mths_ex_med', 'chargeoff_within_12_mths', 'tax_liens']

loan = loan.drop(drop_columnlist,axis=1)
loan.shape
loan.pub_rec_bankruptcies.value_counts()
loan=loan[~loan.pub_rec_bankruptcies.isnull()]
missing = round(100*(loan.isnull().sum()/len(loan.id)), 2) 

missing[missing != 0]
loan=loan[~loan.emp_title.isnull()]

loan=loan[~loan.emp_length.isnull()]
loan.shape
missing = round(100*(loan.isnull().sum()/len(loan.id)), 2) 

missing[missing != 0]
loan=loan[~loan.title.isnull()]

loan=loan[~loan.revol_util.isnull()]

loan=loan[~loan.last_pymnt_d.isnull()]
loan.shape
missing =round(100*(loan.isnull().sum()/len(loan.id)), 2) 

missing[missing != 0]
loan.to_csv('clean_loan.csv', encoding='utf-8', index=False)
clean_loan = loan[:]
clean_loan.nunique().sort_values()
columns_tobe_dropped = ['id','member_id','funded_amnt','funded_amnt_inv','pymnt_plan','url','zip_code','initial_list_status','policy_code','application_type','acc_now_delinq','delinq_amnt',]

clean_loan= clean_loan.drop(columns_tobe_dropped,axis=1)
clean_loan.shape
clean_loan.total_pymnt.value_counts().tail()
clean_loan.total_rec_late_fee.value_counts().tail()
clean_loan.collection_recovery_fee.value_counts().tail()
clean_loan.total_pymnt= round(clean_loan.total_pymnt,2)

clean_loan.total_rec_late_fee= round(clean_loan.total_rec_late_fee,2)

clean_loan.collection_recovery_fee= round(clean_loan.collection_recovery_fee,2)
datetime_colmns=['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d']

clean_loan[datetime_colmns].info()
clean_loan.issue_d = pd.to_datetime(clean_loan.issue_d, format='%b-%y')

clean_loan.earliest_cr_line = pd.to_datetime(clean_loan.earliest_cr_line, format='%b-%y')

clean_loan.last_pymnt_d = pd.to_datetime(clean_loan.last_pymnt_d, format='%b-%y')

clean_loan.last_credit_pull_d = pd.to_datetime(clean_loan.last_credit_pull_d, format='%b-%y')
clean_loan[datetime_colmns].info()
clean_loan=clean_loan.drop_duplicates()

clean_loan.shape
clean_loan.int_rate.describe()
clean_loan.revol_util.describe()
clean_loan['int_rate'] = clean_loan['int_rate'].str.strip('%').astype('float')

clean_loan['revol_util'] = clean_loan['revol_util'].str.strip('%').astype('float')
clean_loan[['int_rate','revol_util']].info()
clean_loan.emp_length.value_counts()
emp_length_dict = {

    '< 1 year' : 0,

    '1 year' : 1,

    '2 years' : 2,

    '3 years' : 3,

    '4 years' : 4,

    '5 years' : 5,

    '6 years' : 6,

    '7 years' : 7,

    '8 years' : 8,

    '9 years' : 9,

    '10+ years' : 10

}
clean_loan = clean_loan.replace({"emp_length": emp_length_dict })
clean_loan.emp_length.value_counts()
clean_loan.term.value_counts()
clean_loan['term'] = clean_loan.term.apply(lambda x: x.split()[0])

clean_loan.term.value_counts()
clean_loan['earliest_cr_line_month'] = clean_loan['earliest_cr_line'].dt.month

clean_loan['earliest_cr_line_year'] = clean_loan['earliest_cr_line'].dt.year

len(clean_loan[clean_loan['earliest_cr_line_year'] > 2011 ])
clean_loan[clean_loan['earliest_cr_line_year'] > 2011 ]['earliest_cr_line_year'].unique()
clean_loan.loc[clean_loan['earliest_cr_line_year'] > 2011 , 'earliest_cr_line_year'] = clean_loan['earliest_cr_line_year'] - 100

clean_loan.groupby('earliest_cr_line_year').loan_amnt.count()
clean_loan['issue_d_month'] = clean_loan['issue_d'].dt.month

clean_loan['issue_d_year'] = clean_loan['issue_d'].dt.year
clean_loan.to_csv('final_loan.csv', encoding='utf-8', index=False)
final_loan = clean_loan[:]

final_loan.shape
final_loan.loan_status.value_counts()
final_loan = final_loan[final_loan['loan_status'].isin(['Fully Paid','Charged Off'])]

final_loan.shape
## Found this solution to show labels in bar plots from https://stackoverflow.com/a/48372659 and edited



def showLabels(ax, d=None):

    plt.margins(0.2, 0.2)

    rects = ax.patches

    i = 0

    locs, labels = plt.xticks() 

    counts = {}

    if not d is None:

        for key, value in d.items():

            counts[str(key)] = value



    # For each bar: Place a label

    for rect in rects:

        # Get X and Y placement of label from rect.

        y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2



        # Number of points between bar and label. Change to your liking.

        space = 5

        # Vertical alignment for positive values

        va = 'bottom'



        # If value of bar is negative: Place label below bar

        if y_value < 0:

            # Invert space to place label below

            space *= -1

            # Vertically align label at top

            va = 'top'



        # Use Y value as label and format number with one decimal place

        if d is None:

            label = "{:.1f}%".format(y_value)

        else:

            try:

                label = "{:.1f}%".format(y_value) + "\nof " + str(counts[str(labels[i].get_text())])

            except:

                label = "{:.1f}%".format(y_value)

        

        i = i+1



        # Create annotation

        plt.annotate(

            label,                      # Use `label` as label

            (x_value, y_value),         # Place label at end of the bar

            xytext=(0, space),          # Vertically shift label by `space`

            textcoords="offset points", # Interpret `xytext` as offset in points

            ha='center',                # Horizontally center label

            va=va)                      # Vertically align label differently for

                                        # positive and negative values.
# This function plots a given column buckets against loan_status (default = 'Charged Off')

# The plots are in percentages 

# (absolute numbers do not make sense -> category values can have very different absolute numbers)

# We want to see what are the chances of some category leading to loan default

# Absolute numbers are also printed to assess level of confidence in a % value. 



def plotLoanStatus(dataframe, by, loanstatus='Charged Off'):

    grp = dataframe.groupby(['loan_status',by])[by].count()

    cnt = dataframe.groupby(by)[by].count()

    #print(grp)

    percentages = grp.unstack() * 100 / cnt.T

    #print(percentages)

    ax = percentages.loc[loanstatus].plot.bar(color=sns.color_palette('husl', 16))

    ax.set_ylabel('% of loans ' + loanstatus)

    showLabels(ax, grp[loanstatus].to_dict())

    plt.margins(0.2, 0.2)

    plt.tight_layout()

    return ax
# This method plots a distribution of target column, and its boxplot against loan_status column



def plot_distribution(dataframe, col):

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)

    ax = sns.distplot(dataframe[col])

    plt.subplot(1, 2, 2)

    sns.boxplot(x=dataframe[col], y=dataframe['loan_status'], data=dataframe)

    plt.show()
# This method prints two plots side by side 

# Left one is percentage of a categorical variable in the entire dataset 

# Right one is percentage for Charged Off loans 

# Significant changes in percentage from left to right can indicate a value of interest



def plot_percentages(dataframe, by, sortbyindex=False):

    plt.subplot(1, 2, 1)

    values = (final_loan[by].value_counts(normalize=True)*100)

    if sortbyindex:

        values = values.sort_index()

    ax = values.plot.bar(color=sns.color_palette('husl', 16))

    ax.set_ylabel('% in dataset', fontsize=16)

    ax.set_xlabel(by, fontsize=12)

    showLabels(ax)

    plt.subplot(1, 2, 2)

    values = (final_loan.loc[final_loan['loan_status']=='Charged Off'][by].value_counts(normalize=True)*100)

    if sortbyindex:

        values = values.sort_index()

    ax = values.plot.bar(color=sns.color_palette('husl', 16))

    ax.set_ylabel('% in Charged Off loans', fontsize=16)

    showLabels(ax)
(final_loan['grade'].value_counts(normalize=True)*100).sort_index()
plt.figure(figsize=(5,7))

ax = final_loan.groupby('loan_status').loan_amnt.count().plot.bar()

showLabels(ax)

plt.show()
print("%.2f" % (final_loan.loc[final_loan['loan_status'] == 'Charged Off'].loan_status.count() * 100/len(final_loan)))
plt.figure(figsize=(5,7))

ax = (final_loan.groupby('loan_status').total_pymnt.sum() * 100 / final_loan.groupby('loan_status').loan_amnt.sum()).plot.bar()

ax.set_ylabel('% loan recovered', fontsize=16)

plt.margins(0.2, 0.2)

showLabels(ax)
plot_distribution(final_loan, 'loan_amnt')
final_loan.groupby('loan_status')['loan_amnt'].describe()
#Create Derived categorical variable

final_loan['loan_amnt_bin'] = pd.cut(final_loan['loan_amnt'], 

                                      [x for x in range(0, 36000, 5000)], labels=[str(x)+'-'+str(x+5)+'k' for x in range (0, 35, 5)])
plotLoanStatus(final_loan, 'loan_amnt_bin')
def categoricalBarPlots(df, arr):

    rows = int(len(arr)/2)

    for idx, val in enumerate(arr, start=1):

        plt.subplot(rows, 2, idx)

        ax = df.groupby(val).loan_amnt.count().plot.bar(color=sns.color_palette('husl', 16))

        showLabels(ax)



    plt.tight_layout()
plt.figure(figsize=(15,15))



categoricalBarPlots(final_loan, ['home_ownership', 'term', 'verification_status', 'purpose', 'grade', 'pub_rec_bankruptcies'])
plt.figure(figsize=(10,5))

plot_percentages(final_loan, 'term')
plt.figure(figsize=(15,5))

plot_percentages(final_loan, 'purpose')
plt.figure(figsize=(10, 5))

plotLoanStatus(final_loan, 'purpose')
plt.figure(figsize=(7,5))

plotLoanStatus(final_loan, 'pub_rec_bankruptcies')
final_loan.int_rate.describe()
plt.figure(figsize=(15,5))

plot_distribution(final_loan, 'int_rate')
final_loan.groupby('loan_status')['int_rate'].describe()
final_loan['interest_rate_buckets'] = round(final_loan['int_rate'])
plt.figure(figsize=(12,5))

plotLoanStatus(final_loan, 'interest_rate_buckets')
final_loan.installment.describe()
plt.figure(figsize=(15,5))

plot_distribution(final_loan, 'installment')
final_loan.grade.value_counts()
final_loan.sub_grade.value_counts(normalize=True).head()
plt.figure(figsize=(10,5))

sns.countplot(final_loan['sub_grade'], order=sorted(final_loan.sub_grade.unique()))

plt.show()
plt.figure(figsize=(15,5))

plot_percentages(final_loan, 'grade', True)
plt.figure(figsize=(10,5))

plotLoanStatus(final_loan, 'grade')
fig, ax = plt.subplots(figsize=(10,5))

final_loan.boxplot(column=['int_rate'],by='grade', ax=ax, rot=90)
# top 10 employee

top_10_emp_title = final_loan.emp_title.value_counts(normalize=False).head(10)
plt.figure(figsize=(10,5))

a=sns.barplot(x=top_10_emp_title.index, y=top_10_emp_title.values)

a.set_ylabel('Count of emp_title')

plt.show()
plotLoanStatus(final_loan[final_loan['emp_title'].isin(top_10_emp_title.index.values)], 'emp_title')
final_loan.emp_length.value_counts(normalize=True)
plt.figure(figsize=(10,5))

sns.countplot(final_loan['emp_length'], order=sorted(final_loan.emp_length.unique()))

plt.show()
plt.figure(figsize=(15, 5))

plot_percentages(final_loan, 'emp_length')
plt.figure(figsize=(10,5))

plot_percentages(final_loan, 'home_ownership')
final_loan.annual_inc.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
# Let's get rid of outliers to analyze annual income. 

# Keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

dataframe = final_loan[np.abs(final_loan.annual_inc-final_loan.annual_inc.mean()) <= (3*final_loan.annual_inc.std())]
dataframe.annual_inc.describe()
plt.figure(figsize=(15, 5))

sns.distplot(final_loan['annual_inc'], hist_kws={'log':False})

plt.xticks(np.arange(0, 260000, 20000))

plt.show()
#Create Derived categorical variable

final_loan['income_bin'] = (final_loan['annual_inc']/20000).astype(int)
plt.figure(figsize=(10,5))

ax = plotLoanStatus(final_loan.loc[final_loan['income_bin']<21], 'income_bin')

ax.set_xticklabels([(str(int(x.get_text())*10)+'-'+str(int(x.get_text())*10+10)+'k') for x in ax.get_xticklabels()])
final_loan.verification_status.value_counts()
plt.figure(figsize=(20,5))

plot_percentages(final_loan, 'verification_status', True)
plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)

sns.countplot(final_loan['issue_d_year'], order=sorted(final_loan.issue_d_year.unique()))



plt.subplot(1, 3, 2)

sns.countplot(final_loan['issue_d_month'], order=sorted(final_loan.issue_d_month.unique()))



#Fraction of loans charged off and fully Paid

plt.subplot(1, 3, 3)

plotLoanStatus(final_loan, 'issue_d_year')



plt.show()
plt.figure(figsize=(10, 5))

plotLoanStatus(final_loan, 'issue_d_month')
final_loan.title.describe()
final_loan.title.value_counts().head(10)
final_loan = final_loan.drop('title',axis =1 )
final_loan.addr_state.value_counts(normalize=True).head(10)
plt.figure(figsize=(30,5))

plt.subplot(1, 2, 1)

sns.countplot(final_loan['addr_state'], order=sorted(final_loan.addr_state.unique()))



#Fraction of loans charged off and fully Paid

charge_off_count = final_loan.groupby('addr_state')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']  

Fully_paid_count = final_loan.groupby('addr_state')['loan_status'].value_counts(normalize=True).loc[:,'Fully Paid']  



plt.figure(figsize=(30,10))

plt.subplot(2, 2, 1)

a=sns.barplot(x=charge_off_count.index, y=charge_off_count.values)

a.set_ylabel('portion of Loans Charged-off')



plt.show()
final_loan.dti.describe()
plt.figure(figsize=(10,5))



plot_distribution(final_loan, 'dti')
# Create derived variable 

final_loan['dti_bin'] = pd.cut(final_loan['dti'], [0,5,10,15,20,25,30], labels=['0-5','5-10','10-15','15-20','20-25','25-30'])

plt.figure(figsize=(10,5))

plotLoanStatus(final_loan, 'dti_bin')
final_loan.delinq_2yrs.value_counts(normalize=True)
plt.figure(figsize=(10,5))

plotLoanStatus(final_loan, 'delinq_2yrs')
final_loan.earliest_cr_line_year.value_counts(normalize=True).head()
plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)



sns.distplot(final_loan['earliest_cr_line_year'])



plt.figure(figsize=(40,10))

plt.subplot(2, 2, 1)

plotLoanStatus(final_loan.loc[final_loan['earliest_cr_line_year'] > 1969], 'earliest_cr_line_year')

plt.show()
final_loan.inq_last_6mths.value_counts(normalize=True)
plt.figure(figsize=(10,10))

plt.subplot(2, 1, 1)

c=sns.countplot(final_loan['inq_last_6mths'], order=sorted(final_loan.delinq_2yrs.unique()))

c.set_yscale('log')



plt.subplot(2, 1, 2)

plotLoanStatus(final_loan, 'inq_last_6mths')

plt.show()
final_loan.open_acc.describe()
plt.figure(figsize=(15,5))

plot_distribution(final_loan, 'open_acc')

plt.show()
final_loan.groupby('loan_status')['open_acc'].describe()
final_loan.pub_rec.value_counts(normalize=True)
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

c=sns.countplot(final_loan['pub_rec'], order=sorted(final_loan.delinq_2yrs.unique()))

c.set_yscale('log')



plt.subplot(1, 2, 2)

plotLoanStatus(final_loan, 'pub_rec')

plt.show()
final_loan.revol_bal.describe()
# keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

final_loan = final_loan[np.abs(final_loan.revol_bal-final_loan.revol_bal.mean()) <= (3*final_loan.revol_bal.std())]

final_loan['revol_bal_log'] = final_loan['revol_bal'].apply(lambda x : np.log(x+1))
plt.figure(figsize=(15,5))



plt.subplot(1, 2, 1)

sns.distplot(final_loan['revol_bal'])



plt.subplot(1, 2, 2)

sns.boxplot(x=final_loan['revol_bal'], y=final_loan['loan_status'], data=final_loan)



plt.show()



plt.subplot(2, 1, 1)

sns.boxplot(x=final_loan['revol_bal'], data=final_loan)

plt.show()
final_loan.groupby('loan_status')['revol_bal'].describe()
final_loan.revol_util.describe()
fig, ax = plt.subplots(figsize=(5,5))

final_loan.boxplot(column=['revol_util'],by='loan_status', ax=ax, rot=90)
final_loan['revol_util_bin'] = round(final_loan['revol_util']/5)
plt.figure(figsize=(14,5))

ax = plotLoanStatus(final_loan, 'revol_util_bin')

ax.set_xticklabels([(str(float(x.get_text())*5)+'%') for x in ax.get_xticklabels()])

plt.show()
final_loan.total_acc.describe()
plt.figure(figsize=(15,5))



plot_distribution(final_loan, 'total_acc')
final_loan.out_prncp.value_counts()
final_loan.groupby('loan_status')['out_prncp'].describe()
final_loan = final_loan.drop('out_prncp',axis=1)
final_loan.out_prncp_inv.value_counts()
final_loan.groupby('loan_status')['out_prncp_inv'].describe()
final_loan = final_loan.drop('out_prncp_inv',axis=1)
final_loan.total_pymnt.describe()
# keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

final_loan = final_loan[np.abs(final_loan.total_pymnt-final_loan.total_pymnt.mean()) <= (3*final_loan.total_pymnt.std())]

final_loan['total_pymnt'] = final_loan['total_pymnt'].apply(lambda x : np.log(x))
plt.figure(figsize=(15,5))

plot_distribution(final_loan, 'total_pymnt')

plt.show()
final_loan = final_loan.drop('total_pymnt',axis=1)
final_loan.last_pymnt_d.value_counts().head()
final_loan['last_pymnt_d_month']= final_loan['last_pymnt_d'].dt.month

final_loan['last_pymnt_d_year']= final_loan['last_pymnt_d'].dt.year
plt.figure(figsize=(10,10))

plt.subplot(2, 1, 1)



sns.distplot(final_loan['last_pymnt_d_year'])



plt.subplot(2, 1, 2)

plotLoanStatus(final_loan, 'last_pymnt_d_year')

plt.show()
final_loan.last_pymnt_amnt.describe()
final_loan['last_pymnt_amnt_log'] = final_loan['last_pymnt_amnt'].apply(lambda x : np.log(x+1))
plt.figure(figsize=(15,5))



plot_distribution(final_loan, 'last_pymnt_amnt_log')

plt.show()
final_loan.last_credit_pull_d.value_counts().head()
final_loan['last_credit_pull_d_month']= final_loan['last_credit_pull_d'].dt.month

final_loan['last_credit_pull_d_year']= final_loan['last_credit_pull_d'].dt.year
final_loan.last_credit_pull_d_year.value_counts(normalize=True)
plt.figure(figsize=(10,5))

sns.distplot(final_loan['last_credit_pull_d_year'])

plt.show()
final_loan['ratio'] = final_loan['loan_amnt'] * 10 / final_loan['annual_inc']

sns.distplot(final_loan['ratio'])
final_loan['ratio_bin'] = (final_loan['ratio'].astype(int)) * 10

plt.figure(figsize=(7,5))

plotLoanStatus(final_loan, 'ratio_bin')
plt.figure(figsize=(10,5))

final_loan.groupby('issue_d_year').loan_amnt.count().plot(kind='line', fontsize=7)

plt.show()
plt.figure(figsize=(10,5))

final_loan.groupby('issue_d_year').loan_amnt.mean().plot(kind='line', fontsize=7)

plt.show()
sns.jointplot('annual_inc', 'loan_amnt', final_loan.loc[final_loan['annual_inc']<260000])

plt.show()
final_loan.boxplot(column='loan_amnt', by='grade')

plt.show()
final_loan.loc[final_loan['annual_inc']<260000].boxplot(column='annual_inc', by='grade')

plt.show()
sns.barplot(x='verification_status', y='loan_amnt', hue="loan_status", data=final_loan, estimator=np.mean)
final_loan.boxplot(column='int_rate', by='term')

plt.show()
final_loan.boxplot(column='int_rate', by='sub_grade',figsize=(10,5))

plt.show()
final_loan.boxplot(column='int_rate', by='loan_status',figsize=(7,5))
sns.barplot(x='grade', y='loan_amnt', hue="term", data=final_loan, estimator=np.mean)

plt.show()
sns.barplot(x='term', y='loan_amnt', hue="loan_status", data=clean_loan, estimator=np.mean)

plt.show()
sns.barplot(x='grade', y='revol_util', hue="loan_status", data=final_loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(20,5))

sns.barplot(x='addr_state', y='loan_amnt', hue='loan_status',data=final_loan, estimator=np.mean)

plt.show()
sns.jointplot('revol_util', 'int_rate', final_loan)
plt.figure(figsize=(20,5))

final_loan.boxplot(column='revol_util', by='grade',figsize=(10,5))

plt.show()
corr=final_loan.corr()

corr['loan_amnt']
plt.figure(figsize=(15,15))

cont_var= ['loan_amnt', 'int_rate', 'installment',

       'emp_length', 'annual_inc',

       'dti', 'delinq_2yrs', 'earliest_cr_line',

       'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',

       'total_acc', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d',

       'pub_rec_bankruptcies']

corr = final_loan[cont_var].corr()

sns.heatmap(corr, annot=True, center=0.5)
sns.jointplot('revol_util', 'loan_amnt', final_loan)
sns.jointplot('int_rate', 'loan_amnt', final_loan)
final_loan['dti_bin']

final_loan.groupby('dti_bin').int_rate.mean()
plt.figure(figsize=(20,5))

sns.barplot(x='dti_bin', y='open_acc', hue='loan_status',data=final_loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(20,5))

sns.barplot(x='delinq_2yrs', y='loan_amnt', hue='grade',data=final_loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(20,5))

sns.barplot(x='delinq_2yrs', y='int_rate', hue='loan_status',data=final_loan, estimator=np.mean)

plt.show()
sns.jointplot('loan_amnt', 'int_rate', final_loan.loc[final_loan.pub_rec_bankruptcies > 0])
sns.jointplot('loan_amnt', 'int_rate', final_loan.loc[final_loan.pub_rec > 0])
final_loan[['pub_rec', 'pub_rec_bankruptcies']].corr()
sns.jointplot('dti', 'int_rate', final_loan)