import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

%matplotlib inline
df=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.head()
df.info()
df.isnull().sum()
df.fillna(df.mean(),inplace=True)
df.isnull().sum()
df.dropna(how='any',inplace=True)
df.info()
df.head()
df=df.drop(['Loan_ID'],axis=1)
a=df.select_dtypes('object').columns[:-1]
a
df1=pd.DataFrame()
for i in a:

    df2=pd.get_dummies(df[i],drop_first=True)

    df1=pd.concat([df2,df1],axis=1)

    df=df.drop(i,axis=1)
df=pd.concat([df1,df],axis=1)
df.head()
df.info()
x=df.iloc[:,:-1]

y=df.iloc[:,-1]
x.head()
y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))