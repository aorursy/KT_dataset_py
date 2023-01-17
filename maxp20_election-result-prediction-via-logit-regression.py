# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


from sklearn.model_selection import train_test_split



import seaborn as sns

# sns.set()

sns.set(rc={'figure.figsize':(11.7,8.27)})
dat = pd.read_csv('/kaggle/input/electionfinance/CandidateSummaryAction1.csv',

                 parse_dates=['cov_sta_dat','cov_end_dat'])



# Convert dollars to floats

def convdollars2float(dfcol):

    val = dfcol.str.replace('$','').str.replace(',','').str.replace('(','-').str.replace(')','').astype('float32')

    return val



for colname in ['cas_on_han_beg_of_per','cas_on_han_clo_of_per','net_con','net_ope_exp','deb_owe_by_com','deb_owe_to_com']:

    dat[colname] = convdollars2float(dat[colname])



# Convert winner column to boolean

dat['winner'] = dat['winner'].apply(lambda val: float(int(val=='Y')))



dat.head(10)


# Add column: time length cov_end_dat - cov_sta_dat

dat['cov_dat_len_days'] = (dat['cov_end_dat'] - dat['cov_sta_dat']).dt.days

# Drop where this column has negative value

dat = dat[ dat['cov_dat_len_days'] >= 0 ]

# Add column: number of competitiors



# First, construct lookup table of number of candidates running for each district

num_comp_lookup = dat.groupby(['can_off_sta','can_off_dis'])['can_id'].count().to_dict()



# Then perform lookup on this table to append number of competitors

def fcn(row):

    key = (row.can_off_sta,row.can_off_dis)

    if key in num_comp_lookup:

        return num_comp_lookup[key]

#     else:

#        # Handling NaNs:

#        num_competitors = 0

dat['num_comp'] = dat.apply(lambda row: fcn(row), axis=1)



# Drop rows where number of competitors couldn't be calculated (>0) and where uncontested (>1)

dat = dat[ dat['num_comp'] > 1 ]

# Add column: fraction of spend for this district



# First, construct lookup table of total spend for this district

total_net_con_for_district = dat.groupby(['can_off_sta','can_off_dis'])['net_con'].sum().to_dict()



# Add column giving this total of net contributions

def fcn(row):

    key = (row.can_off_sta,row.can_off_dis)

    if key in num_comp_lookup:

        return total_net_con_for_district[key]

#     else:

#        # Handling NaNs:

#        num_competitors = 0

dat['total_net_con_for_district'] = dat.apply(lambda row: fcn(row), axis=1)



# Calculate the fraction

dat['fraction_net_con_for_district'] = dat['net_con'] / dat['total_net_con_for_district']
# # Add column: tot_comp per vote

# dat['net_con_per_vote'] = dat['net_con']/dat['votes']
dat.head()
# Winning probability is function of:

# Number of competitors: num_comp

# Number of days campaigning: cov_dat_len_days ???TBC TODO is this what this column refers to???

# Fraction of total money available/spent by the candidate: fraction_net_con_for_district ???TBC TODO is this what this column refers to???



# COLS_TO_REGRESS = ['num_comp','cov_dat_len_days','fraction_net_con_for_district']

# COLS_TO_REGRESS = ['fraction_net_con_for_district']

COLS_TO_REGRESS = ['num_comp','cov_dat_len_days']



_ = dat[['winner']+COLS_TO_REGRESS].dropna()



y = _['winner']

X = _[COLS_TO_REGRESS]



import statsmodels.api as sm

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
# ROC Curve



import matplotlib.pyplot as plt 

plt.rc("font", size=14)



from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])



plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
# Classification report

from sklearn.metrics import classification_report

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))
# Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
# Add column of predictions alongside each row

dat_grp_preds = dat[['can_id']+COLS_TO_REGRESS].dropna(axis=0)



dat_grp_preds['winner_prediction'] = logreg.predict( dat_grp_preds[COLS_TO_REGRESS] )



dat_grp_preds = dat_grp_preds.set_index('can_id')



dat_grp_preds.head()

# Join main data with predictions data on can_id

_ = dat.set_index('can_id')

_ = _.join(dat_grp_preds['winner_prediction'])
# Group the data by district

# dat_grp = dat.groupby(['can_off_sta','can_off_dis','can_inc_cha_ope_sea']).sum()

_grp = _.set_index(['can_off_sta','can_off_dis']).sort_values(by=['can_off_sta','can_off_dis'])

_grp