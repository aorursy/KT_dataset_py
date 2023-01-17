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
credit_data = pd.read_csv('/kaggle/input/credit-card-approval-prediction/credit_record.csv')
credit_data.head()
application_data = pd.read_csv('/kaggle/input/credit-card-approval-prediction/application_record.csv')
application_data.head()
application_data.sort_values(by = ['ID'])
application_data.describe()
application_data.isna().sum()
len(application_data)
application_data.info()
import seaborn as sns
import matplotlib.pyplot as plt
application_data.CODE_GENDER.value_counts()
#sns.countplot(application_data.CODE_GENDER)
sns.countplot(x='CODE_GENDER', data = application_data, hue = 'FLAG_OWN_CAR')
sns.countplot(x='CODE_GENDER', data = application_data, hue = 'FLAG_OWN_REALTY')
sns.countplot(y = 'CNT_CHILDREN', data = application_data)
print(application_data.CNT_CHILDREN.value_counts())
application_data.AMT_INCOME_TOTAL.describe()
sns.boxplot(x = 'AMT_INCOME_TOTAL', data = application_data)
plt.figure(figsize = (10,8))
sns.countplot(x = 'NAME_INCOME_TYPE', data = application_data, hue='CODE_GENDER')
plt.figure(figsize = (8,6))
sns.countplot(y = 'NAME_EDUCATION_TYPE', data = application_data, hue='CODE_GENDER')
plt.figure(figsize = (8,6))
sns.countplot(y = 'NAME_FAMILY_STATUS', data = application_data,hue='CODE_GENDER')
plt.figure(figsize = (8,6))
sns.countplot(y = 'NAME_HOUSING_TYPE', data = application_data,hue='CODE_GENDER')
application_data['AGE'] = round(application_data.DAYS_BIRTH *(-1/365))
application_data['YEARS_EMPLOYED'] = round(application_data.DAYS_EMPLOYED *(-1/365))
application_data
application_data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'],1, inplace = True)
application_data.head()
sns.distplot(application_data.AGE, bins = 10)
application_data.AGE.hist(grid = True, bins = 5)
application_data.YEARS_EMPLOYED[application_data.YEARS_EMPLOYED <0]= -1
application_data.YEARS_EMPLOYED.value_counts()
application_data.info()
application_data.YEARS_EMPLOYED[application_data.YEARS_EMPLOYED == -1]
application_data.YEARS_EMPLOYED.hist(grid = True, bins = 10)
application_data.OCCUPATION_TYPE.value_counts()
application_data['OCCUPATION_TYPE'][application_data['OCCUPATION_TYPE'].isna()]='Not mentioned'
application_data['OCCUPATION_TYPE']
plt.figure(figsize = (8,6))
sns.countplot(y = 'OCCUPATION_TYPE', data = application_data,hue='CODE_GENDER')
application_data.columns
application_data.head()
import pandas as pd
dummy_data = pd.get_dummies(application_data[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']], prefix_sep = '_', drop_first= True)
dummy_data.head()
application_data = application_data.join(dummy_data)
application_data = application_data.drop(['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'], axis = 1)
application_data.CNT_CHILDREN.value_counts()
application_data.CNT_CHILDREN[application_data.CNT_CHILDREN>=3]=3
application_data.CNT_CHILDREN.value_counts()
application_data.CNT_FAM_MEMBERS.value_counts()
application_data.CNT_FAM_MEMBERS[application_data.CNT_FAM_MEMBERS>=4]=4
application_data[['CNT_CHILDREN', 'CNT_FAM_MEMBERS']] = application_data[['CNT_CHILDREN', 'CNT_FAM_MEMBERS']].astype('str')
import pandas as pd
dummy_data = pd.get_dummies(application_data[['CNT_CHILDREN', 'CNT_FAM_MEMBERS']], prefix_sep = '_', drop_first= True)
dummy_data.head()
application_data = application_data.join(dummy_data)
application_data.head()
application_data = application_data.drop(['CNT_CHILDREN', 'CNT_FAM_MEMBERS'], axis = 1)
application_data.head()
application_data.YEARS_EMPLOYED.value_counts()
application_data['YEARS_EMPLOYED'][application_data['YEARS_EMPLOYED']<0.0]='un_emp'


application_data['YEARS_EMPLOYED'].value_counts()
application_data['YEARS_EMPLOYED'][application_data['YEARS_EMPLOYED']!='un_emp']='emp'
application_data['YEARS_EMPLOYED'].value_counts()
import pandas as pd
dummies = pd.get_dummies(application_data[['YEARS_EMPLOYED']],prefix_sep = '_', drop_first= True)
dummies.head()
application_data.join(dummies)
application_data = application_data.join(dummies)
application_data.head()
application_data.drop(['YEARS_EMPLOYED'],1, inplace = True)
application_data.head()
application_data['AMT_INCOME_TOTAL']=(application_data['AMT_INCOME_TOTAL']- application_data['AMT_INCOME_TOTAL'].min())/(application_data['AMT_INCOME_TOTAL'].max()- application_data['AMT_INCOME_TOTAL'].min())
application_data['AGE']=(application_data['AGE']- application_data['AGE'].min())/(application_data['AGE'].max()- application_data['AGE'].min())
application_data.head()
credit_data
print(len(credit_data['ID'].unique()))
credit_data['ID'].unique()
no_loan_data = credit_data[credit_data['STATUS']=='X']
no_loan_data.head()
print(len(no_loan_data['ID'].unique()))
no_loan_data['ID'].unique()
loan_data = credit_data[credit_data["STATUS"]!= 'X']
loan_data.head()
print(len(loan_data['ID'].unique()))
loan_data['ID'].unique()
exempted_data = loan_data[(loan_data['MONTHS_BALANCE']==0) | (loan_data['MONTHS_BALANCE']==-1) | (loan_data['MONTHS_BALANCE']==-2) | (loan_data['MONTHS_BALANCE']==-3)]
exempted_data.head()
len(exempted_data)
non_exempted_data = loan_data[loan_data['MONTHS_BALANCE']<=-4]
non_exempted_data.head()
fraud_data = non_exempted_data[non_exempted_data.STATUS !='C']
fraud_data
fraud_data_compile = pd.DataFrame(fraud_data['ID'].unique(), columns = ['ID'])
fraud_data_compile['Fraud_Pred']=1
fraud_data_compile.head()
list(fraud_data_compile['ID'])
least_fraud = non_exempted_data[non_exempted_data.STATUS =='C']
#least_fraud_compile = pd.DataFrame(least_fraud['ID'].unique(), columns = ['ID'])
#least_fraud_compile.head()
least_fraud['ID'].unique()
new_credit_third_month= list( set(list(least_fraud['ID'])) - set(list(fraud_data_compile['ID'])))
new_credit_third_month
new_credit_third_month = pd.DataFrame(new_credit_third_month, columns = ['ID'])
new_credit_third_month['Fraud_Pred'] = 0
new_credit_third_month.head()
fraud_data_compile_new = pd.concat([fraud_data_compile, new_credit_third_month], axis= 0)
fraud_data_compile_new
exempted_list = list(exempted_data['ID'].unique())
exempted_list
new_holders = list(set(exempted_list)-set(list(fraud_data_compile['ID'])))
new_holders = pd.DataFrame(new_holders, columns = ['ID'])
new_holders['Fraud_Pred']=0
new_holders.head()
fraud_data_compile_new_1 = pd.concat([fraud_data_compile, new_holders], axis= 0)
fraud_data_compile_new_1
#fraud_data_compile_new[fraud_data_compile_new.Fraud_Pred ==0]
len(fraud_data_compile_new_1)
len(fraud_data_compile_new_1['ID'].unique())
len(no_loan_data['ID'].unique())
new_cust_no_loan = list(set(list(no_loan_data['ID'].unique())) - set((list(fraud_data_compile_new_1['ID']))))
new_cust_no_loan = pd.DataFrame(new_cust_no_loan, columns = ['ID'])
new_cust_no_loan['Fraud_Pred'] = 0
new_cust_no_loan.head()
fraud_data_compile_new_2 = pd.concat([fraud_data_compile_new_1, new_cust_no_loan], axis= 0)
fraud_data_compile_new_2

fraud_data_compile_new_2.sort_values(by = ['ID'])
application_data
application_data.sort_values(by = ['ID'])
application_data[application_data['ID']==7603224]
application_data['ID'].value_counts()
X = pd.merge(application_data,fraud_data_compile_new_2,on= 'ID', how = 'left')
X
X.shape  
X['Fraud_Pred'].isna().sum()
X = X.dropna()
X.head()
X.shape
X= X.drop(['ID'],1)
X
from sklearn.model_selection import train_test_split
features = X.drop('Fraud_Pred', axis =1)
labels = X['Fraud_Pred']
features.head()
len(labels)
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.4, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size= 0.5, random_state=42)
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))
print(len(X_val), len(y_val))
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)
gb = GradientBoostingClassifier()
parameters = {
    'n_estimators' : [5,50,200,250],
    'max_depth'  : [1,3,5,7,9],
    'learning_rate': [0.01,0.1,1,10,100]    
}
cv = GridSearchCV(gb, parameters, cv=5)
cv.fit(X_train, y_train.values.ravel())
cv
cv.best_params_
gb = GradientBoostingClassifier(learning_rate= 0.1,max_depth= 9,n_estimators= 50)
gb.fit(X_train, y_train)
from sklearn.metrics import classification_report,confusion_matrix
val_predict = gb.predict(X_val)
print(classification_report(y_val, val_predict))
print(confusion_matrix(y_val, val_predict))
from sklearn.metrics import classification_report,confusion_matrix
test_predict = gb.predict(X_test)
print(classification_report(y_test, test_predict))
print(confusion_matrix(y_test, test_predict))
