import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_data = pd.read_csv('../input/train_AV3.csv')
train_data.head()
train_data.info()
train_data.describe()
sns.countplot(x='Gender',data=train_data)
train_data['Gender'][train_data['Gender'].isnull()]='Male'
sns.countplot(x='Married',data=train_data)
train_data['Married'][train_data['Married'].isnull()]='Yes'
train_data['LoanAmount'][train_data['LoanAmount'].isnull()]= train_data['LoanAmount'].mean()
sns.countplot(x='Loan_Amount_Term',data=train_data)
train_data['Loan_Amount_Term'][train_data['Loan_Amount_Term'].isnull()]='360'
sns.countplot(x='Self_Employed',data=train_data)
train_data['Self_Employed'][train_data['Self_Employed'].isnull()]='No'
sns.countplot(x='Credit_History',data=train_data)
train_data['Credit_History'][train_data['Credit_History'].isnull()]='1'
train_data.info()
sns.countplot(x='Dependents',data=train_data)
train_data['Dependents'][train_data['Dependents'].isnull()]='0'
train_data.loc[train_data.Dependents=='3+','Dependents']= 4
train_data.tail()
train_data.loc[train_data.Loan_Status=='N','Loan_Status']= 0
train_data.loc[train_data.Loan_Status=='Y','Loan_Status']=1
train_data.loc[train_data.Gender=='Male','Gender']= 0
train_data.loc[train_data.Gender=='Female','Gender']=1
train_data.loc[train_data.Married=='No','Married']= 0
train_data.loc[train_data.Married=='Yes','Married']=1
train_data.loc[train_data.Education=='Graduate','Education']= 0
train_data.loc[train_data.Education=='Not Graduate','Education']=1
train_data.loc[train_data.Self_Employed=='No','Self_Employed']= 0
train_data.loc[train_data.Self_Employed=='Yes','Self_Employed']=1
property_area= pd.get_dummies(train_data['Property_Area'],drop_first=True)

train_data= pd.concat([train_data,property_area],axis=1)
train_data.head()
from sklearn.cross_validation import train_test_split

X= train_data.drop(['Loan_ID','Property_Area','Loan_Status'],axis=1)
y = train_data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
prediction= logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))
data_test= pd.read_csv('../input/test_AV3.csv')
data_test.head()
data_test.info()
sns.countplot('Gender',data=data_test)
data_test['Gender'][data_test['Gender'].isnull()]='Male'
sns.countplot('Dependents',data=data_test)
data_test['Dependents'][data_test['Dependents'].isnull()]=0
data_test.loc[data_test.Dependents=='3+','Dependents']= 4
sns.countplot('Self_Employed',data=data_test)
data_test['Self_Employed'][data_test['Self_Employed'].isnull()]='No'
sns.countplot('Loan_Amount_Term',data=data_test)
data_test['Loan_Amount_Term'][data_test['Loan_Amount_Term'].isnull()]=360
sns.countplot('Credit_History',data=data_test)
data_test['Credit_History'][data_test['Credit_History'].isnull()]=1
sns.countplot('Property_Area',data=data_test)
data_test['Property_Area'][data_test['Property_Area'].isnull()]='Urban'
data_test.head()
data_test['LoanAmount'][data_test['LoanAmount'].isnull()]= data_test['LoanAmount'].mean()
data_test.loc[data_test.Gender=='Male','Gender']= 0
data_test.loc[data_test.Gender=='Female','Gender']=1
data_test.loc[data_test.Married=='No','Married']= 0
data_test.loc[data_test.Married=='Yes','Married']=1
data_test.loc[data_test.Education=='Graduate','Education']= 0
data_test.loc[data_test.Education=='Not Graduate','Education']=1
data_test.loc[data_test.Self_Employed=='No','Self_Employed']= 0
data_test.loc[data_test.Self_Employed=='Yes','Self_Employed']=1
property_area= pd.get_dummies(data_test['Property_Area'],drop_first=True)
data_test = pd.concat([data_test,property_area],axis=1)
X_data_test= data_test.drop(['Loan_ID','Property_Area'],axis=1)
X_data_test.head()
data_test['Loan_Status']= logmodel.predict(X_data_test)
data_frame=data_test[['Loan_ID','Loan_Status']]
data_frame.loc[data_frame.Loan_Status==0,'Loan_Status']='N'
data_frame.loc[data_frame.Loan_Status==1,'Loan_Status']='Y'
data_frame.head()
data_frame.to_csv('Loan Predictions Submission.csv',index=0)
