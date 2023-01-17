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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

df=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()
df.describe()
df.describe(include=['object', 'bool'])
df.Contract.value_counts()
df.Churn.value_counts()
df.Dependents.value_counts()
df.Churn.value_counts(normalize=True)
df.dtypes
#I am changing Churn to 1 and 0 for analysis purpose

df['Churn']=df['Churn'].replace('No',0)

df['Churn']=df['Churn'].replace('Yes',1)
#I am changing Seniorcitizon as yes or no

df['SeniorCitizen']=df['SeniorCitizen'].replace(1,'Yes')

df['SeniorCitizen']=df['SeniorCitizen'].replace(0,'No')
# Here Totalcharges actually is integer but it showing object so i converted into integer

df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.head()

pd.set_option('display.max_columns', None)
df.head()
df.shape


df.isnull().sum()
#Defining categorical variables

categorical_features=df.select_dtypes(include=[object])
categorical_features.columns
# Removing unwanted coloumns

df.drop(['customerID'],axis=1,inplace=True)
df.TotalCharges.isnull().sum()
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())
df.TotalCharges.isnull().sum()
df.dtypes
## below are manually done encoding
df['gender']=df['gender'].replace('Male',1)

df['gender']=df['gender'].replace('Female',0)
df['Partner']=df['Partner'].replace('No',0)

df['Partner']=df['Partner'].replace('Yes',1)
df['Dependents']=df['Churn'].replace('No',0)

df['Dependents']=df['Dependents'].replace('Yes',1)
df['SeniorCitizen']=df['SeniorCitizen'].replace('No',0)

df['SeniorCitizen']=df['SeniorCitizen'].replace('Yes',1)
df['PhoneService']=df['PhoneService'].replace('No',0)

df['PhoneService']=df['PhoneService'].replace('Yes',1)
df.head()
df.dtypes
### One Hot Encoding by ceating dummies
from sklearn.preprocessing import OneHotEncoder
oe=OneHotEncoder()
final_df=pd.get_dummies(columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'],data=df)
final_df.head()
final_df.shape
y=final_df['Churn']
X=final_df.drop(['Churn'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
X_train.shape
X_test.shape
y_test.shape
y_train.shape
X.shape
final_df.dtypes
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_train,y_train)
lr_predict=lr.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm=confusion_matrix(y_test,lr_predict)
cm
metrics.accuracy_score(y_test,lr_predict)
metrics.roc_auc_score(y_test,lr_predict)
metrics.classification_report(y_test,lr_predict)
metrics.f1_score(y_test,lr_predict)
print('Actual:', y_test.values[0:25])

print('Predicted:', lr_predict[0:25])
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='entropy',n_estimators=1000,max_depth=100,oob_score=True,random_state=42)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
rf_predict=rf.predict(X_test)
metrics.confusion_matrix(y_test,rf_predict)
metrics.accuracy_score(y_test,rf_predict)
metrics.f1_score(y_test,rf_predict)
import pandas as pd

feature_importances1 = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances1