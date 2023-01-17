import numpy as np

import pandas as pd
bank=pd.read_excel('../input/loan.xlsx')
print(bank.head())

print(bank.shape)
del bank['id']

del bank['member_id']

del bank['emp_title']

del bank['issue_d']

del bank['url']

del bank['desc']

del bank['title']

del bank['zip_code']

del bank['addr_state']

del bank['earliest_cr_line']

del bank['last_pymnt_d']

del bank['next_pymnt_d']

del bank['last_credit_pull_d']

del bank['mths_since_last_major_derog']

del bank['annual_inc_joint']

del bank['dti_joint']

del bank['verification_status_joint']

del bank['tot_coll_amt']

del bank['tot_cur_bal']

del bank['open_acc_6m']

del bank['open_il_6m']

del bank['open_il_12m']

del bank['open_il_24m']

del bank['mths_since_rcnt_il']

del bank['total_bal_il']

del bank['il_util']

del bank['open_rv_12m']

del bank['open_rv_24m']

del bank['max_bal_bc']

del bank['all_util']

del bank['total_rev_hi_lim']

del bank['inq_fi']

del bank['total_cu_tl']

del bank['inq_last_12m']

del bank['acc_open_past_24mths']

del bank['avg_cur_bal']

del bank['bc_open_to_buy']

del bank['bc_util']

del bank['mo_sin_old_il_acct']

del bank['mo_sin_old_rev_tl_op']

del bank['mo_sin_rcnt_rev_tl_op']

del bank['mo_sin_rcnt_tl']

del bank['mort_acc']

del bank['mths_since_recent_bc']

del bank['mths_since_recent_bc_dlq']

del bank['mths_since_recent_inq']

del bank['mths_since_recent_revol_delinq']

del bank['num_accts_ever_120_pd']

del bank['num_actv_bc_tl']

del bank['num_actv_rev_tl']

del bank['num_bc_sats']

del bank['num_bc_tl']

del bank['num_il_tl']

del bank['num_op_rev_tl']

del bank['num_rev_accts']

del bank['num_rev_tl_bal_gt_0']

del bank['num_sats']

del bank['num_tl_120dpd_2m']

del bank['num_tl_30dpd']

del bank['num_tl_90g_dpd_24m']

del bank['num_tl_op_past_12m']

del bank['pct_tl_nvr_dlq']

del bank['percent_bc_gt_75']

del bank['tot_hi_cred_lim']

del bank['total_bal_ex_mort']

del bank['total_bc_limit']

del bank['total_il_high_credit_limit']
bank.shape
bank.describe()
del bank['collections_12_mths_ex_med']

del bank['acc_now_delinq']

del bank['chargeoff_within_12_mths']

del bank['delinq_amnt']

del bank['tax_liens']
bank.shape
bank.head()
bank.isna().sum()
def classify(s):

    if(s=='10+ years'):

        return 10

    elif(s=='1 year'):

        return 1

    elif(s=='2 years'):

        return 2

    elif(s=='3 years'):

        return 3

    elif(s=='4 years'):

        return 4

    elif(s=='5 years'):

        return 5

    elif(s=='6 years'):

        return 6

    elif(s=='7 years'):

        return 7

    elif(s=='8 years'):

        return 8

    elif(s=='9 years'):

        return 9

    else:

        return 0
bank['emp_length']=bank.emp_length.apply(classify)
bank.mths_since_last_delinq.fillna(0,inplace=True)
del bank['mths_since_last_record']
bank.pub_rec_bankruptcies.fillna(bank.pub_rec_bankruptcies.mean(),inplace=True)
bank.revol_util.fillna(0,inplace=True)
bank
bank.term.unique()
def term(s):

    if(s=='36 months'):

        return 36

    else:

        return 60
bank['term']=bank.term.apply(term)
bank.grade.unique()
def grade(s):

    if(s=='A'):

        return 1

    elif(s=='B'):

        return 2

    elif(s=='C'):

        return 3

    elif(s=='D'):

        return 4

    elif(s=='E'):

        return 5

    elif(s=='F'):

        return 6

    else:

        return 7
bank['grade']=bank.grade.apply(grade)
del bank['sub_grade']
bank.home_ownership.unique()
def home(s):

    if(s=='RENT'):

        return 1

    elif(s=='OWN'):

        return 2

    elif(s=='MORTGAGE'):

        return 3

    elif(s=='OTHER'):

        return 4

    else:

        return 5
bank['home_ownership']=bank.home_ownership.apply(home)
bank.application_type.unique()
del bank['application_type']
bank.columns
bank.verification_status.unique()
def verification(s):

    if(s=='Verified'):

        return 1

    elif(s=='Source Verified'):

        return 2

    else:

        return 3
bank['verification_status']=bank.verification_status.apply(verification)
y=bank['loan_status']
y.unique()
del bank['loan_status']
def loan_status(s):

    if(s=='Fully Paid'):

        return 1

    elif(s=='Charged Off'):

        return 2

    else:

        return 3
y=y.apply(loan_status)
y.describe()
bank.pymnt_plan.unique()
del bank['pymnt_plan']
bank.purpose.unique()
def purpose(s):

    if(s=='other'):

        return 2

    else:

        return 1
bank['purpose']=bank.purpose.apply(purpose)
bank.initial_list_status.unique()
del bank['initial_list_status']
bank.describe()
bank.shape
from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(bank,y,test_size=0.3)
print(x_train.shape)

print(x_test.shape)
x_dev,x_test1,y_dev,y_test1=model_selection.train_test_split(x_test,y_test,test_size=0.5)
print(x_dev.shape)

print(x_test1.shape)
from xgboost.sklearn import XGBClassifier
clf=XGBClassifier(random_state=1,n_jobs=8)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test1)
clf.score(x_test1,y_test1)
learning=[0.1,0.5,0.01,0.05,0.001]

for i in range(0,5):

    clf1=XGBClassifier(random_state=1,learning_rate=learning[i])

    clf1.fit(x_train,y_train)

    print(clf.score(x_dev,y_dev))
learning=[0.1,0.5,0.01,0.05,0.001]

for i in range(0,5):

    clf1=XGBClassifier(random_state=1,learning_rate=learning[i],booster='gblinear')

    clf1.fit(x_train,y_train)

    print(clf.score(x_dev,y_dev))
print('Test accuracy score')

clf.score(x_test,y_test)
print('Train_accuracy_score')

clf.score(x_train,y_train)
x_train
columns=x_test.columns
import matplotlib.pyplot as plt
x=x_test.values

y=y_test.values
for i in range(0,33):

    plt.scatter(x[0:1000,i],y[0:1000,])

    plt.xlabel(columns[i])

    plt.ylabel("y")

    plt.show()

    plt.scatter(x[2000:3000,i],y[2000:3000,])

    plt.xlabel(columns[i])

    plt.ylabel("y")

    plt.show()

    plt.scatter(x[3000:4000,i],y[3000:4000,])

    plt.xlabel(columns[i])

    plt.ylabel("y")

    plt.show()