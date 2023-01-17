# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import xgboost
train_data= pd.read_csv('../input/train.csv')

test_data= pd.read_csv('../input/test.csv')

train_data['source']= 'train'

test_data['source']='test'

data=pd.concat([test_data,train_data],axis=0)

data2=pd.read_csv('../input/test.csv')
data.describe()
data.isnull()

null_columns=data.columns[data.isnull().any()]

data[null_columns].isnull().sum()



a=pd.DataFrame(data[data['loan_default']==1]['supplier_id'].value_counts())

b=pd.DataFrame(data['supplier_id'].value_counts())

a['index1']=a.index

b['index1']=b.index

c=pd.merge(a,b,on=['index1'],how='outer')

c.fillna(value=0,inplace=True)

c['Percentage_tot']=c['supplier_id_y']/sum(c['supplier_id_y'])*100

c['Percentage_1']=c['supplier_id_x']/c['supplier_id_y']

dict_supplier_per=dict(zip(c.index1,c.Percentage_1))

data['supplier_id_perc']= data['supplier_id'].map(dict_supplier_per)


a=pd.DataFrame(data[data['loan_default']==1]['branch_id'].value_counts())

b=pd.DataFrame(data['branch_id'].value_counts())

a['index1']=a.index

b['index1']=b.index

c=pd.merge(a,b,on=['index1'],how='outer')

c.fillna(value=0,inplace=True)

c['Percentage_tot']=c['branch_id_y']/sum(c['branch_id_y'])*100

c['Percentage_1']=c['branch_id_x']/c['branch_id_y']

dict_branch_per=dict(zip(c.index1,c.Percentage_1))

data['branch_id_perc']= data['branch_id'].map(dict_branch_per)


a=pd.DataFrame(data[data['loan_default']==1]['state_id'].value_counts())

b=pd.DataFrame(data['state_id'].value_counts())

a['index1']=a.index

b['index1']=b.index

c=pd.merge(a,b,on=['index1'],how='outer')

c.fillna(value=0,inplace=True)

c['Percentage_tot']=c['state_id_y']/sum(c['state_id_y'])*100

c['Percentage_1']=c['state_id_x']/c['state_id_y']

dict_state_per=dict(zip(c.index1,c.Percentage_1))

data['state_id_perc']= data['state_id'].map(dict_state_per)


a=pd.DataFrame(data[data['loan_default']==1]['employee_code_id'].value_counts())

b=pd.DataFrame(data['employee_code_id'].value_counts())

a['index1']=a.index

b['index1']=b.index

c=pd.merge(a,b,on=['index1'],how='outer')

c.fillna(value=0,inplace=True)

c['Percentage_tot']=c['employee_code_id_y']/sum(c['employee_code_id_y'])*100

c['Percentage_1']=c['employee_code_id_x']/c['employee_code_id_y']

dict_Empl_ID_per=dict(zip(c.index1,c.Percentage_1))

data['Emp_id_perc']= data['employee_code_id'].map(dict_Empl_ID_per)
def aggregate_risk(x):

    if 'Risk' in x:

        return x.split('-')[1]

    elif 'No Bureau History' in x:

        return 'No History'

    else:

        return 'No Score'

data['cns_clubbed']=data.cns_score_description.apply(aggregate_risk)

data=pd.get_dummies(data,columns=['cns_clubbed'],drop_first=True)

data['age']=(pd.to_datetime(data.disbursal_date)-pd.to_datetime(data.date_of_birth))/np.timedelta64(1,'Y')
def gettmonths(x):

    yrs=x.split('yrs')[0]

    mon=x.split()[1].split('mon')[0]

    new_tenure=(int(yrs)*12)+int(mon)

    return new_tenure




data['diff_avgFirstloan']=data.credit_history_length.apply(gettmonths)-data.avg_account_age.apply(gettmonths)


cols=[ 'disbursed_amt', 'asset_cost', 'ltv','loan_default',

        'employment_type_Self employed',

       'pri_no_of_accounts', 'pri_active_accounts', 'pri_overdue_accounts',

       'pri_current_balance', 'pri_sanctioned_amt', 'pri_disbursed_amt',

        'new_account_last_6_months',

       'default_in_last_6_months',

       'no_of_inquiries', 'supplier_id_perc', 'branch_id_perc',

       'state_id_perc', 'Emp_id_perc', 'age', 'diff_avgFirstloan',

       'cns_clubbed_Low Risk', 'cns_clubbed_Medium Risk',

       'cns_clubbed_No History', 'cns_clubbed_No Score',

       'cns_clubbed_Very High Risk', 'cns_clubbed_Very Low Risk','source']
#data['employment_type'].fillna('unk',inplace=True)

#data=pd.get_dummies(data,columns=['employment_type'],drop_first=True)

data_missing=data[data['employment_type'].isnull()]

data_not_missing=data[data['employment_type'].notnull()]

cols_m=[ 'disbursed_amt', 'asset_cost', 'ltv',

        'employment_type','loan_default',

       'pri_no_of_accounts', 'pri_active_accounts', 'pri_overdue_accounts',

       'pri_current_balance', 'pri_sanctioned_amt', 'pri_disbursed_amt',

        'new_account_last_6_months',

       'default_in_last_6_months',

       'no_of_inquiries', 'supplier_id_perc', 'branch_id_perc',

       'state_id_perc', 'Emp_id_perc', 'age', 'diff_avgFirstloan',

       'cns_clubbed_Low Risk', 'cns_clubbed_Medium Risk',

       'cns_clubbed_No History', 'cns_clubbed_No Score',

       'cns_clubbed_Very High Risk', 'cns_clubbed_Very Low Risk','source']

data_missing=data_missing[cols_m]

data_not_missing= data_not_missing[cols_m]

data_missing.drop(['employment_type'],axis=1,inplace=True)

x=data_not_missing[['disbursed_amt', 'asset_cost', 'ltv', 

       'pri_no_of_accounts', 'pri_active_accounts', 'pri_overdue_accounts',

       'pri_current_balance', 'pri_sanctioned_amt', 'pri_disbursed_amt',

        'new_account_last_6_months',

       'default_in_last_6_months',

       'no_of_inquiries', 'supplier_id_perc', 'branch_id_perc',

       'state_id_perc', 'Emp_id_perc', 'age', 'diff_avgFirstloan',

       'cns_clubbed_Low Risk', 'cns_clubbed_Medium Risk',

       'cns_clubbed_No History', 'cns_clubbed_No Score',

       'cns_clubbed_Very High Risk', 'cns_clubbed_Very Low Risk']]

y=data_not_missing['employment_type']

from sklearn.model_selection import train_test_split

x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.1,random_state=0)

clf=xgboost.XGBClassifier()



clf.fit(x_train,y_train)
data_missing2= data_missing[['disbursed_amt', 'asset_cost', 'ltv', 

       'pri_no_of_accounts', 'pri_active_accounts', 'pri_overdue_accounts',

       'pri_current_balance', 'pri_sanctioned_amt', 'pri_disbursed_amt',

        'new_account_last_6_months',

       'default_in_last_6_months',

       'no_of_inquiries', 'supplier_id_perc', 'branch_id_perc',

       'state_id_perc', 'Emp_id_perc', 'age', 'diff_avgFirstloan',

       'cns_clubbed_Low Risk', 'cns_clubbed_Medium Risk',

       'cns_clubbed_No History', 'cns_clubbed_No Score',

       'cns_clubbed_Very High Risk', 'cns_clubbed_Very Low Risk']]


pred_val= clf.predict(data_missing2)
pred_val
data_missing2['employment_type']=pred_val

data_missing2['loan_default']=data_missing['loan_default']

data_missing2['source']=data_missing['source']
data3=pd.concat([data_missing2,data_not_missing],axis=0)
data3=pd.get_dummies(data3,columns=['employment_type'],drop_first=True)
data=data3


train_data= data[data['source']=='train']

test_data= data[data['source']=='test']

test_data=test_data[cols]

data=train_data[cols]



# data.dropna(inplace=True,axis=1)

from sklearn.preprocessing import StandardScaler

cols_to_scale = ['disbursed_amt', 'asset_cost', 'ltv', 'pri_no_of_accounts',

       'pri_active_accounts', 'pri_overdue_accounts', 'pri_current_balance',

       'pri_sanctioned_amt', 'pri_disbursed_amt', 'new_account_last_6_months',

       'default_in_last_6_months', 'no_of_inquiries',

       'supplier_id_perc', 'branch_id_perc', 'state_id_perc', 'Emp_id_perc',

       'age', 'diff_avgFirstloan']

sc= StandardScaler()



data[cols_to_scale] = sc.fit_transform(data[cols_to_scale])

test_data[cols_to_scale] = sc.transform(test_data[cols_to_scale])



x=data[['disbursed_amt', 'asset_cost', 'ltv', 'employment_type_Self employed'

        , 'pri_no_of_accounts', 'pri_active_accounts',

        'pri_overdue_accounts', 'pri_current_balance', 'pri_sanctioned_amt',

        'pri_disbursed_amt', 'new_account_last_6_months',

        'default_in_last_6_months', 'no_of_inquiries',

        'supplier_id_perc', 'branch_id_perc', 'state_id_perc', 'Emp_id_perc',

        'age', 'diff_avgFirstloan', 'cns_clubbed_Low Risk',

        'cns_clubbed_Medium Risk', 'cns_clubbed_No History',

        'cns_clubbed_No Score', 'cns_clubbed_Very High Risk',

        'cns_clubbed_Very Low Risk']]

y=data['loan_default']

from sklearn.model_selection import train_test_split

x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=0)


#clf=xgboost.XGBClassifier()

#clf.fit(x_train,y_train)
from sklearn.model_selection import GridSearchCV

parameters = {#when use hyperthread, xgboost may become slower,

              'learning_rate': [0.001,0.01,0.03], #so called `eta` value,

              'max_depth': [3, 4, 5],

              'n_estimators': [100,200,300]}

xgb1= xgboost.XGBClassifier()

clf = GridSearchCV(xgb1,parameters,cv = 2,n_jobs = 5,verbose=True)

clf.fit(x_train,y_train)


pred_val= clf.predict_proba(x_validation)
from sklearn import metrics



fpr, tpr, thresholds=metrics.roc_curve(y_validation, pred_val[:,1])

metrics.auc(fpr, tpr)
data.loc[0,'ltv']
test_data=test_data[['disbursed_amt', 'asset_cost', 'ltv', 'employment_type_Self employed',

         'pri_no_of_accounts', 'pri_active_accounts',

        'pri_overdue_accounts', 'pri_current_balance', 'pri_sanctioned_amt',

        'pri_disbursed_amt', 'new_account_last_6_months',

        'default_in_last_6_months', 'no_of_inquiries',

        'supplier_id_perc', 'branch_id_perc', 'state_id_perc', 'Emp_id_perc',

        'age', 'diff_avgFirstloan', 'cns_clubbed_Low Risk',

        'cns_clubbed_Medium Risk', 'cns_clubbed_No History',

        'cns_clubbed_No Score', 'cns_clubbed_Very High Risk',

        'cns_clubbed_Very Low Risk']]



pred_val= clf.predict_proba(test_data)
unique_id_test=data2.unique_id

submss=pd.DataFrame(unique_id_test)

submss['loan_default']=pred_val[:,1]

submss.rename(columns={'unique_id':'UniqueID'},inplace=True)
submss.to_csv('submss_prop2.csv',index=False)