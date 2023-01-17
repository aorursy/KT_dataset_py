#Import neccesary libraries

import pandas as pd

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import os

import re

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier

import lightgbm as lgb

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,roc_auc_score, roc_curve, auc, precision_score

%matplotlib inline

matplotlib.rcParams.update({'font.size': 20})

from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTETomek
#Read Data

train = pd.read_csv('../input/train.csv', infer_datetime_format=True)

test = pd.read_csv('../input/test.csv', infer_datetime_format=True)

# des = pd.read_excel('./Data Dictionary.xlsx')
#Check train test 

print(train.shape)

print(test.shape)



#Drop duplicates

# test.drop_duplicates()

# No duplicates in data

# print(test.shape)

# train.isnull().sum()
# test.isna().sum()
# train.info()
train.head()
train.nunique()
train.loan_default.value_counts()
#Drop null values as this data is too sparse with 0 as most values

# train.dropna(inplace=True)

# test.dropna(inplace=True)

print(train.shape)

print(test.shape)

# print(train.isnull().sum().sum())



# print(test.isnull().sum().sum())
train.describe()
test.head()
def credit_risk(df):

    d1=[]

    d2=[]

    for i in df:

        p = i.split("-")

        if len(p) == 1:

            d1.append(p[0])

            d2.append('unknown')

        else:

            d1.append(p[1])

            d2.append(p[0])



    return d1,d2



def calc_number_of_ids(row):

#     print(type(row), row.size)

    return sum(row[['Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag',

       'Passport_flag']])



def check_pri_installment(row):

    if row['PRIMARY.INSTAL.AMT']<=1:

        return 0

    else:

        return row['PRIMARY.INSTAL.AMT']

    

def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()

risk_map = {'No Bureau History Available':-1, 

              'Not Scored: No Activity seen on the customer (Inactive)':-1,

              'Not Scored: Sufficient History Not Available':-1,

              'Not Scored: No Updates available in last 36 months':-1,

              'Not Scored: Only a Guarantor':-1,

              'Not Scored: More than 50 active Accounts found':-1,

              'Not Scored: Not Enough Info available on the customer':-1,

              'Very Low Risk':4,

              'Low Risk':3,

              'Medium Risk':2, 

              'High Risk':1,

              'Very High Risk':0}



sub_risk = {'unknown':-1, 'I':5, 'L':2, 'A':13, 'D':10, 'M':1, 'B':12, 'C':11, 'E':9, 'H':6, 'F':8, 'K':3,

       'G':7, 'J':4}

employment_map = {'Self employed':0, 'Salaried':1,np.nan:-1}
def features_engineering(df):

    print('feature engineering started')

    df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format = "%d-%m-%y",infer_datetime_format=True)

    df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'], format = "%d-%m-%y",infer_datetime_format=True)

    now = pd.Timestamp('now')

    df['Age'] = (now - df['Date.of.Birth']).astype('<m8[Y]').astype(int)

    age_mean = int(df[df['Age']>0]['Age'].mean())

    df.loc[:,'age'] = df['Age'].apply(lambda x: x if x>0 else age_mean)

    df['disbursal_months_passed'] = ((now - df['DisbursalDate'])/np.timedelta64(1,'M')).astype(int)

    df['average_act_age_in_months'] = df['AVERAGE.ACCT.AGE'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))

    df['credit_history_length_in_months'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))

    df['number_of_0'] = (df == 0).astype(int).sum(axis=1)

    

    df.loc[:,'credit_risk'],df.loc[:,'credit_risk_grade']  = credit_risk(df["PERFORM_CNS.SCORE.DESCRIPTION"])

    

    df.loc[:, 'loan_to_asset_ratio'] = df['disbursed_amount'] /df['asset_cost']

    df.loc[:,'no_of_accts'] = df['PRI.NO.OF.ACCTS'] + df['SEC.NO.OF.ACCTS']



    df.loc[:,'pri_inactive_accts'] = df['PRI.NO.OF.ACCTS'] - df['PRI.ACTIVE.ACCTS']

    df.loc[:,'sec_inactive_accts'] = df['SEC.NO.OF.ACCTS'] - df['SEC.ACTIVE.ACCTS']

    df.loc[:,'tot_inactive_accts'] = df['pri_inactive_accts'] + df['sec_inactive_accts']

    df.loc[:,'tot_overdue_accts'] = df['PRI.OVERDUE.ACCTS'] + df['SEC.OVERDUE.ACCTS']

    df.loc[:,'tot_current_balance'] = df['PRI.CURRENT.BALANCE'] + df['SEC.CURRENT.BALANCE']

    df.loc[:,'tot_sanctioned_amount'] = df['PRI.SANCTIONED.AMOUNT'] + df['SEC.SANCTIONED.AMOUNT']

    df.loc[:,'tot_disbursed_amount'] = df['PRI.DISBURSED.AMOUNT'] + df['SEC.DISBURSED.AMOUNT']

    df.loc[:,'tot_installment'] = df['PRIMARY.INSTAL.AMT'] + df['SEC.INSTAL.AMT']

    df.loc[:,'bal_disburse_ratio'] = np.round((1+df['tot_disbursed_amount'])/(1+df['tot_current_balance']),2)

    df.loc[:,'pri_tenure'] = (df['PRI.DISBURSED.AMOUNT']/( df['PRIMARY.INSTAL.AMT']+1)).astype(int)

    df.loc[:,'sec_tenure'] = (df['SEC.DISBURSED.AMOUNT']/(df['SEC.INSTAL.AMT']+1)).astype(int)

#     df.loc[:,'tenure_to_age_ratio'] =  np.round((df['pri_tenure']/12)/df['age'],2)

    df.loc[:,'disburse_to_sactioned_ratio'] =  np.round((df['tot_disbursed_amount']+1)/(1+df['tot_sanctioned_amount']),2)

    df.loc[:,'active_to_inactive_act_ratio'] =  np.round((df['no_of_accts']+1)/(1+df['tot_inactive_accts']),2)

    print('done')

#     df.loc[:,'']

    return df
def label_data(df):

    print('labeling started')

    df.loc[:,'credit_risk_label'] = df['credit_risk'].apply(lambda x: risk_map[x])

    df.loc[:,'sub_risk_label'] = df['credit_risk_grade'].apply(lambda x: sub_risk[x])

    df.loc[:,'employment_label'] = df['Employment.Type'].apply(lambda x: employment_map[x])

    print('labeling done')

    return df
def data_correction(df):

    print('invalid data handling started')

    #Many customers have invalid date of birth, so immute invalid data with mean age

    df.loc[:,'PRI.CURRENT.BALANCE'] = df['PRI.CURRENT.BALANCE'].apply(lambda x: 0 if x<0 else x)

    df.loc[:,'SEC.CURRENT.BALANCE'] = df['SEC.CURRENT.BALANCE'].apply(lambda x: 0 if x<0 else x)

    

    #loan that do not have current pricipal outstanding should have 0 primary installment

    df.loc[:,'new_pri_installment']= df.apply(lambda x : check_pri_installment(x),axis=1)

    print('done')

    return df
def prepare_data(df):

    df = data_correction(df)

    df = features_engineering(df)

    df = label_data(df)



    return df

    
#Prepare training and test data

train_data = prepare_data(train)

train_data = train_data[train_data['number_of_0']<=25]

test_data = prepare_data(test)
train_data[train_data['number_of_0']>=20]['number_of_0'].value_counts()
train_data.columns
to_drop = ['UniqueID', 'ltv', 'branch_id',

       'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Date.of.Birth',

       'Employment.Type', 'DisbursalDate', 'State_ID', 'Employee_code_ID',

       'MobileNo_Avl_Flag', 'PRIMARY.INSTAL.AMT',

       'PERFORM_CNS.SCORE.DESCRIPTION',

       'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 

       'loan_default', 'Age',  'credit_risk', 'credit_risk_grade',

       ]

features = ['disbursed_amount', 'asset_cost',

            'Aadhar_flag', 'PAN_flag',

       'PERFORM_CNS.SCORE',

             'PRI.ACTIVE.ACCTS',

       'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',

       'PRI.DISBURSED.AMOUNT',  'SEC.ACTIVE.ACCTS',

       'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',

       'SEC.DISBURSED.AMOUNT',  'SEC.INSTAL.AMT',

       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',

            'NO.OF_INQUIRIES','disbursal_months_passed',

       'average_act_age_in_months', 'credit_history_length_in_months',

       'number_of_0','loan_to_asset_ratio', 'no_of_accts', 'pri_inactive_accts',

       'sec_inactive_accts', 'tot_inactive_accts', 'tot_overdue_accts',

       'tot_current_balance', 'tot_sanctioned_amount', 'tot_disbursed_amount',

       'tot_installment', 'bal_disburse_ratio', 'pri_tenure', 'sec_tenure',

       'credit_risk_label',

       'employment_label', 'age', 'new_pri_installment'

           ]



print(train_data.shape)

print(test_data.shape)
from sklearn.preprocessing import  RobustScaler

# std_scaler = StandardScaler()

# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()



scaled_training = train_data.copy()

scaled_testing = test_data.copy()





scaled_training[features] = rob_scaler.fit_transform(scaled_training[features])

scaled_testing[features] = rob_scaler.fit_transform(scaled_testing[features])



y = scaled_training.loan_default

X = scaled_training[features]

# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27,stratify=y)

print(X_train.shape, y_train.shape)

print(X_test.shape,y_test.shape)

sm = SMOTE(random_state=2)

X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

print(X_train.shape, y_train.shape)
#  Prepare data for modeling

# Separate input features and target

# plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

#Fitting the PCA algorithm with our Data

pca = PCA(n_components=7).fit(X)

X = pca.fit_transform(X)

X = pd.DataFrame(X, columns = ['p1','p2','p3','p4','p5','p6','p7'])

test_df = pd.DataFrame(pca.fit_transform(scaled_testing[features]), columns = ['p1','p2','p3','p4','p5','p6','p7'])

#Plotting the Cumulative Summation of the Explained Variance

plt.figure(figsize=(15,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_), 'r-')

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('L& T Loan Dataset Explained Variance')

plt.show()
X.shape
# # setting up testing and training sets

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27,stratify=y)
train.columns
# test['number_of_0'].value_counts()

# Drop row having more than 50 % data as 0

# train_data = train[train['number_of_0']<=20]

# test_data = test[test['number_of_0']<=20]

# train_data = train.copy()

# test_data = test.copy()
f, axes = plt.subplots(1,2 , figsize=(20, 7), sharex=True)

# sns.despine(left=True)

sns.distplot(train['asset_cost'],kde = False, color="b", ax=axes[0])

sns.distplot(test['asset_cost'],kde = False, color="r", ax=axes[1])





# plt.pyplot.setp(axes, yticks=[])

# plt.pyplot.tight_layout()
f, axes = plt.subplots(1,2 , figsize=(20, 7), sharex=True)

# sns.despine(left=True)

sns.distplot(train['ltv'],kde = False, color="b", ax=axes[0])

sns.distplot(test['ltv'],kde = False, color="r", ax=axes[1])

# train['ltv'].value_counts()
fig, ax =plt.subplots(1,1,figsize=(4.5, 5))

sns.countplot(train['Aadhar_flag'],  palette="Set3").set_title('Aadhar count')

fig, ax =plt.subplots(1,2,figsize=(10, 5))

# fig, ax =plt.subplots(2,2,figsize=(10, 10))

# sns.countplot(train['Aadhar_flag'], ax=ax[0,0],  palette="Set3")

# sns.countplot(test['Aadhar_flag'], ax=ax[0,1],  palette="Set3")

sns.countplot(train[train['Aadhar_flag']==1]['loan_default'], ax=ax[0], palette="Set3").set_title('Aadhar=1')

sns.countplot(train[train['Aadhar_flag']==0]['loan_default'], ax=ax[1],  palette="Set3").set_title('Aadhar=0')
fig, ax =plt.subplots(1,1,figsize=(4.5, 5))

sns.countplot(train['PAN_flag'],  palette="Set3").set_title('PAN count')

fig, ax =plt.subplots(1,2,figsize=(10, 5))

# sns.countplot(test['PAN_flag'], ax=ax[0,1] ,palette="Set3")

# sns.countplot(train[train['PAN_flag']==1]['loan_default'], ax=ax[0,0],palette="Set3").set_title("LaLaLa")

# sns.countplot(train[train['PAN_flag']==0]['loan_default'], ax=ax[0,1],palette="Set3")

sns.countplot(train[train['PAN_flag']==1]['loan_default'], ax=ax[0],palette="Set3").set_title("PAN=1")

sns.countplot(train[train['PAN_flag']==0]['loan_default'], ax=ax[1],palette="Set3").set_title("PAN=0")
fig, ax =plt.subplots(1,1,figsize=(4.5, 5))

sns.countplot(train['VoterID_flag'],  palette="Set3").set_title('VoterID count')

fig, ax =plt.subplots(1,2,figsize=(10, 5))

# sns.countplot(test['PAN_flag'], ax=ax[0,1] ,palette="Set3")

# sns.countplot(train[train['PAN_flag']==1]['loan_default'], ax=ax[0,0],palette="Set3").set_title("LaLaLa")

# sns.countplot(train[train['PAN_flag']==0]['loan_default'], ax=ax[0,1],palette="Set3")

sns.countplot(train[train['VoterID_flag']==1]['loan_default'], ax=ax[0],palette="Set3").set_title("VoterID=1")

sns.countplot(train[train['VoterID_flag']==0]['loan_default'], ax=ax[1],palette="Set3").set_title("VoterID=0")

#######

# fig, ax =plt.subplots(2,2,figsize=(10, 10))

# sns.countplot(train['VoterID_flag'], ax=ax[0,0])

# sns.countplot(test['VoterID_flag'], ax=ax[0,1])

# sns.countplot(train[train['VoterID_flag']==1]['loan_default'], ax=ax[1,0])

# sns.countplot(train[train['VoterID_flag']==0]['loan_default'], ax=ax[1,1])
fig, ax =plt.subplots(2,2,figsize=(10, 10))

sns.countplot(train['Driving_flag'], ax=ax[0,0])

sns.countplot(test['Driving_flag'], ax=ax[0,1])

sns.countplot(train[train['Driving_flag']==1]['loan_default'], ax=ax[1,0])

sns.countplot(train[train['Driving_flag']==0]['loan_default'], ax=ax[1,1])
fig, ax =plt.subplots(2,2,figsize=(10, 10))

sns.countplot(train['Passport_flag'], ax=ax[0,0])

sns.countplot(test['Passport_flag'], ax=ax[0,1])

sns.countplot(train[train['Passport_flag']==1]['loan_default'], ax=ax[1,0])

sns.countplot(train[train['Passport_flag']==0]['loan_default'], ax=ax[1,1])
def train_model(model):

    # Checking accuracy

    model = model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print('accuracy_score',accuracy_score(y_test, pred))

    print('precision_score',precision_score(y_test, pred))

    print('recall_score',recall_score(y_test, pred))

    print('f1_score',f1_score(y_test, pred))

    print('roc_auc_score',roc_auc_score(y_test, pred))

    # confusion matrix

    print('confusion_matrix')

    print(pd.DataFrame(confusion_matrix(y_test, pred)))

    return model
# Modeling the data as is

# Train model

# xgb = XGBClassifier()



# xgb = train_model(xgb)
# from xgboost import plot_importance

# from matplotlib import pyplot

# # plot feature importance

# plot_importance(xgb)

# pyplot.show()
# train model

rfc = RandomForestClassifier()

rfc = train_model(rfc)

# predict on test set

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=0)

dt_clf = train_model(dt_clf)
d_train = lgb.Dataset(X_train, label=y_train)

params = {}



params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 10

params['min_data'] = 50

params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)
pred=clf.predict(X_test)

for i in range(len(pred)):

    if pred[i]>=.4:       # setting threshold to .5

        pred[i]=1

    else:  

        pred[i]=0

print('accuracy_score',accuracy_score(y_test, pred))

print('recall_score',recall_score(y_test, pred))

print('f1_score',f1_score(y_test, pred))

print('roc_auc_score',roc_auc_score(y_test, pred))

# confusion matrix

print('confusion_matrix')

print(pd.DataFrame(confusion_matrix(y_test, pred)))
# f = ['p1','p2','p3','p4','p5']

# fi = xgb.feature_importances_

# rfi = rfc.feature_importances_

# xgbfi = pd.DataFrame({'features':f,'xgb_importance':fi, 'rf_importance':rfi})

# xgbfi.sort_values(by=['rf_importance'],ascending=False)
# best_parameters = gd_sr.best_params_  

# print(best_parameters)  
# best_result = gd_sr.best_score_  

# print(best_result)  
unique_id = scaled_testing.UniqueID

y_pred_rf = rfc.predict(scaled_testing[features])

submission1 = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_rf})

submission1.head()



# unique_id = testing.UniqueID

# y_pred_rf = xgb.predict(testing.drop(to_drop_test, axis=1))

# submission2 = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_rf})

# submission2.head()



filename = 'submission_rf.csv'



submission1.to_csv(filename,index=False)



print('Saved file: ' + filename)



# filename1 = 'submission_xgb.csv'



# submission2.to_csv(filename1,index=False)



# print('Saved file: ' + filename1)