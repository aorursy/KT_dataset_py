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
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
train
train.drop('Loan_ID',axis=1,inplace=True)
train
train.isnull().sum()
train['LoanAmount']=train['LoanAmount'].fillna(train['LoanAmount'].mean())
train['Credit_History']= train['Credit_History'].fillna(train['Credit_History'].median())
train.isnull().sum()
df=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.dropna(inplace=True)
df.shape
df['Gender'].value_counts()
df['Married'].value_counts()
df['Education'].value_counts()
df['Self_Employed'].value_counts()
df['Property_Area'].value_counts()
plt.boxplot(df['ApplicantIncome'])
plt.boxplot(df['CoapplicantIncome'])
plt.boxplot(df['LoanAmount'])
plt.boxplot(df['Loan_Amount_Term'])
plt.boxplot(df['Credit_History'])
sns.scatterplot(x='Property_Area',y='Loan_Status',data=df)
sns.scatterplot(x='Self_Employed',y='Loan_Status',data=df)
print(pd.crosstab(df['Property_Area'],df['Loan_Status']))
sns.countplot(df['Property_Area'],hue=df['Loan_Status'])
sns.countplot(df['Gender'],hue=df['Loan_Status'])
print(pd.crosstab(df['Married'],df['Loan_Status']))
sns.countplot(df['Married'],hue=df['Loan_Status'])
print(pd.crosstab(df['Self_Employed'],df['Loan_Status']))
sns.countplot(df['Self_Employed'],hue=df['Loan_Status'])
print(pd.crosstab(df['Education'],df['Loan_Status']))
sns.countplot(df['Education'],hue=df['Loan_Status'])
df['Loan_Status'].replace('N',0,inplace=True)
df['Loan_Status'].replace('Y',1,inplace=True)
plt.title('Correlation Matrix')
sns.heatmap(df.corr(),annot=True)
df2=df.drop(labels=['ApplicantIncome'],axis=1)
df2=df2.drop(labels=['CoapplicantIncome'],axis=1)
df2=df2.drop(labels=['LoanAmount'],axis=1)
df2=df2.drop(labels=['Loan_Amount_Term'],axis=1)
df2=df2.drop(labels=['Loan_ID'],axis=1)
df2.head()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder()
df2['Property_Area']=le.fit_transform(df2['Property_Area'])
df2['Dependents']=le.fit_transform(df2['Dependents'])
df2=pd.get_dummies(df2)
df2.dtypes
df2=df2.drop(labels=['Gender_Female'],axis=1)
df2=df2.drop(labels=['Married_No'],axis=1)
df2=df2.drop(labels=['Education_Not Graduate'],axis=1)
df2=df2.drop(labels=['Self_Employed_No'],axis=1)
df2.head()
plt.title('Correlation Matrix')
sns.heatmap(df2.corr(),annot=True)
df2=df2.drop('Self_Employed_Yes',1)
df2=df2.drop('Dependents',1)
df2=df2.drop('Education_Graduate',1)
X=df2.drop('Loan_Status',1)
Y=df2['Loan_Status']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=6)
print('Shape of X_train is: ',x_train.shape)
print('Shape of X_test is: ',x_test.shape)
print('Shape of Y_train is: ',y_train.shape)
print('Shape of y_test is: ',y_test.shape)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
log.score(x_train,y_train)
pred=log.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)
from sklearn import metrics
metrics.confusion_matrix(y_test,pred)
metrics.recall_score(y_test,pred)
metrics.precision_score(y_test,pred)
metrics.f1_score(y_test,pred)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
clf.score(x_train,y_train)
pred1=clf.predict(x_test)
accuracy_score(y_test,pred1)
metrics.confusion_matrix(y_test,pred1)
metrics.f1_score(y_test,pred1)
metrics.recall_score(y_test,pred1)
metrics.precision_score(y_test,pred1)