# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,StratifiedKFold,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score,roc_auc_score,roc_curve,auc
%matplotlib inline

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
train= pd.read_csv('../input/ltfs-av-data/train.csv',infer_datetime_format=True)
test= pd.read_csv('../input/ltfs-av-data/test_bqCt9Pv.csv',infer_datetime_format=True)
train.shape
test.shape
train.info()
train.head()
train.nunique()
train.loan_default.value_counts()
train.isna().sum()
train['Employment.Type'].value_counts()
train.isna().sum()
train.describe()
for i in train.columns:
    print(i,': distinct_value')
    print(train[i].nunique(),': No.of unique items')
    print(train[i].unique())
    print('-'*30)
    print('')
def credit_risk(df):
    d1=[]
    d2=[]
    for i in df:
        p=i.split('-')
        if len(p)==1:
            d1.append(p[0])
            d2.append('unknown')
        else:
            d1.append(p[1])
            d2.append(p[0])
    return d1,d2

def number_of_ids(row):
    return sum(row[['Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag','Passport_flag']])

def check_pri_installment(row):
    if row['PRIMARY.INSTAL.AMT']<=1:
        return 0
    else:
        return row['PRIMARY.INSTAL.AMT']

def plot_2d_space(X,y,label='Classes'):
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
risk_map={'No Bureau History Available':-1,
         'Not Scored: Not Enough Info available on the customer':-1,
         'Not Scored: No Updates available in last 36 months':-1,
         'Not Scored: No Activity seen on the customer (Inactive)':-1,
         'Not Scored: Sufficient History Not Available':-1,
         'Not Scored: Only a Guarantor':-1,
         'Not Scored: More than 50 active Accounts found':-1,
         'Very Low Risk':4,
         'Low Risk':3,
         'Medium Risk':2,
         'High Risk':1,
         'Very High Risk':0}
sub_risk={'unknown':-1,'A':13,'B':12,'C':11,'D':10,'E':9,'F':8,'G':7,'H':6,'I':5,'J':4,'K':3,'L':2,'M':1}
employment_map={'Self employed':0,'Salaried':1,np.nan:-1}
def Feature_Engineering(df):
    df['DisbursalDate']= pd.to_datetime(df['DisbursalDate'],format='%d-%m-%y',infer_datetime_format=True)
    df['Date.of.Birth']= pd.to_datetime(df['Date.of.Birth'],format='%d-%m-%y',infer_datetime_format=True)
    now=pd.Timestamp('now')
    df['Age']=(now- df['Date.of.Birth']).astype('<m8[Y]').astype(int)
    age_mean=int(df[df['Age']>0]['Age'].mean())
    df.loc[:,'age']=df['Age'].apply(lambda x:x if x>0 else age_mean)
    df['disbursal_months_passed']=((now -df['DisbursalDate'])/np.timedelta64(1,'M')).astype(int)
    df['AvgAcctAge_Months']=df['AVERAGE.ACCT.AGE'].apply(lambda x: int(re.findall(r'\d+',x)[0])*12 +int(re.findall(r'\d+',x)[1]))
    df['CreditHistoryLength_Months']=df['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(re.findall(r'\d+',x)[0])*12+int(re.findall(r'\d+',x)[1]))
    df['number_of_zero']=(df==0).astype(int).sum(axis=1)
    df.loc[:,'credit_risk'],df.loc[:,'credit_risk_grade']= credit_risk(df['PERFORM_CNS.SCORE.DESCRIPTION'])
    df.loc[:,'loan_to_asset_ratio']=df['disbursed_amount']/df['asset_cost']
    df.loc[:,'no_of_accounts']=df['PRI.NO.OF.ACCTS']+df['SEC.NO.OF.ACCTS']
    df.loc[:,'pri_inactive_accts']=df['PRI.NO.OF.ACCTS']-df['PRI.ACTIVE.ACCTS']
    df.loc[:,'sec_inactive_accts']=df['SEC.NO.OF.ACCTS']-df['SEC.ACTIVE.ACCTS']
    df.loc[:,'total_inactive_accts']=df['pri_inactive_accts']+df['sec_inactive_accts']
    df.loc[:,'total_overdue_accts']=df['PRI.OVERDUE.ACCTS']+df['SEC.OVERDUE.ACCTS']
    df.loc[:,'total_current_balance']=df['PRI.CURRENT.BALANCE']+df['SEC.CURRENT.BALANCE']
    df.loc[:,'total_sanctioned_amount']=df['PRI.SANCTIONED.AMOUNT']+df['SEC.SANCTIONED.AMOUNT']
    df.loc[:,'total_disbursed_amount']=df['PRI.DISBURSED.AMOUNT']+df['SEC.DISBURSED.AMOUNT']
    df.loc[:,'total_installment']=df['PRIMARY.INSTAL.AMT']+df['SEC.INSTAL.AMT']
    df.loc[:,'bal_disburse_ratio']=np.round((1+df['total_disbursed_amount'])/(1+df['total_current_balance']),2)
    df.loc[:,'pri_tenure']=(df['PRI.DISBURSED.AMOUNT']/(df['PRIMARY.INSTAL.AMT']+1)).astype(int)
    df.loc[:,'sec_tenure']=(df['SEC.DISBURSED.AMOUNT']/(df['SEC.INSTAL.AMT']+1)).astype(int)
    df.loc[:,'disburse_to_sanction_ratio']=np.round((df['total_disbursed_amount']+1)/(1+df['total_sanctioned_amount']),2)
    df.loc[:,'active_to_inactive_ratio']=np.round((df['no_of_accounts']+1)/(1+df['total_inactive_accts']),2)
    
    return df
   



    


def label_data(df):
    df.loc[:,'credit_risk_label']=df.loc[:,'credit_risk'].apply(lambda x: risk_map[x])
    df.loc[:,'sub_risk_label']=df.loc[:,'credit_risk_grade'].apply(lambda x: sub_risk[x])
    df.loc[:,'employment_label']=df.loc[:,'Employment.Type'].apply(lambda x: employment_map[x])
    return df
def data_correction(df):
    df.loc[:,'PRI.CURRENT.BALANCE']=df['PRI.CURRENT.BALANCE'].apply(lambda x:0 if x<0 else x)
    df.loc[:,'SEC.CURRENT.BALANCE']=df['SEC.CURRENT.BALANCE'].apply(lambda x:0 if x<0 else x)
    
    df.loc[:,'new_pri_installment']=df.apply(lambda x: check_pri_installment(x),axis=1)
    return df

def prepare_data(df):
    df=data_correction(df)
    df=Feature_Engineering(df)
    df=label_data(df)
    return df
train_data= prepare_data(train)
train_data=train_data[train_data['number_of_zero']<=25]
test_data=prepare_data(test)
train_data.columns
to_drop=['UniqueID', 'ltv', 'branch_id',
       'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Date.of.Birth',
       'Employment.Type', 'DisbursalDate', 'State_ID', 'Employee_code_ID',
       'MobileNo_Avl_Flag', 'PRIMARY.INSTAL.AMT',
       'PERFORM_CNS.SCORE.DESCRIPTION',
       'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 
       'loan_default', 'Age',  'credit_risk', 'credit_risk_grade',]

Features=['disbursed_amount', 'asset_cost',
            'Aadhar_flag', 'PAN_flag',
       'PERFORM_CNS.SCORE',
             'PRI.ACTIVE.ACCTS',
       'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
       'PRI.DISBURSED.AMOUNT',  'SEC.ACTIVE.ACCTS',
       'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
       'SEC.DISBURSED.AMOUNT',  'SEC.INSTAL.AMT',
       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            'NO.OF_INQUIRIES','disbursal_months_passed',
       'AvgAcctAge_Months', 'CreditHistoryLength_Months',
       'number_of_zero','loan_to_asset_ratio', 'no_of_accounts', 'pri_inactive_accts',
       'sec_inactive_accts', 'total_inactive_accts', 'total_overdue_accts',
       'total_current_balance', 'total_sanctioned_amount', 'total_disbursed_amount',
       'total_installment', 'bal_disburse_ratio', 'pri_tenure', 'sec_tenure',
       'credit_risk_label',
       'employment_label', 'age', 'new_pri_installment'
           ]
train_data.shape
test_data.shape
scaler= RobustScaler()

scaled_train= train_data.copy()
scaled_test= test_data.copy()

scaled_train[Features]=scaler.fit_transform(scaled_train[Features])
scaled_test[Features]=scaler.fit_transform(scaled_test[Features])

X=scaled_train[Features]
y=scaled_train.loan_default
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=30,stratify=y)
X_train.shape
y_train.shape
X_test.shape
smote=SMOTE(random_state=2)
X_train,y_train=smote.fit_sample(X_train,y_train.ravel())
X_train.shape
y_train.shape
def train_model(model):
    model=model.fit(X_train,y_train)
    pred=model.predict(X_test)
    print('accuracy_score',accuracy_score(y_test, pred))
    print('recall_score',recall_score(y_test, pred))
    print('f1_score',f1_score(y_test, pred))
    print('roc_auc_score',roc_auc_score(y_test, pred))
    print('confusion_matrix')
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    return model
lr=LogisticRegression()
rf=RandomForestClassifier()
dtc=DecisionTreeClassifier()
xgb=XGBClassifier()

lr= train_model(lr)
rf=train_model(rf)
dtc=train_model(dtc)
xgb=train_model(xgb)
xgb=XGBClassifier(max_depth=2,n_estimators=1000,eta=0.4)
xgb=train_model(xgb)
unique_id= scaled_test.UniqueID
y_predxgb=xgb.predict(scaled_test[Features])
submission=pd.DataFrame({'UniqueID':unique_id,'loan_default':y_predxgb})
submission.head()
filename= 'submission.csv'
submission.to_csv(filename,index=False)