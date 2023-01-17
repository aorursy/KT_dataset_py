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
df=pd.read_csv('../input/train.csv')

df['Age']=df['Age'].fillna(df['Age'].median())

df2=pd.get_dummies(df['Sex'])

df['female']=df2['female']

df['male']=df2['male']

df.drop('Sex',axis=1,inplace=True)

df2=pd.get_dummies(df['Embarked'])

df['C']=df2['C']

df['Q']=df2['Q']

df['S']=df2['S']

df.drop(['Embarked'],axis=1,inplace=True)

#print(df2.head())

#print(df.info())

#print(df.head())

y_train=df['Survived']

x_train=df.drop(['Survived','Name','Ticket','Cabin'],axis=1)

#print(x_train.info())

x_test=pd.read_csv('../input/test.csv')

x_test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

df2=pd.get_dummies(x_test['Sex'])

x_test['female']=df2['female']

x_test['male']=df2['male']

x_test.drop('Sex',axis=1,inplace=True)

x_test['Fare'].fillna(7.00625,inplace=True)

x_test['Age'].fillna(df['Age'].median(),inplace=True)#*****

df2=pd.get_dummies(x_test['Embarked'])

x_test['C']=df2['C']

x_test['Q']=df2['Q']

x_test['S']=df2['S']

x_test.drop('Embarked',axis=1,inplace=True)

print(x_test.info())

#print(x_test[x_test['Age'].isnull()==True])

#print(df[['Fare','SibSp','Parch']][np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(df['Pclass']==3,df['SibSp']==0),df['Parch']==0),df['S']==1),df['Age']>60),df['male']==1)])



#from sklearn.linear_model import LogisticRegression



import xgboost as xgb

model=xgb.XGBClassifier()

model.fit(x_train,y_train)

y_test=model.predict(x_test)

ans=pd.DataFrame({'PassengerId':x_test['PassengerId'],'Survived':y_test})

print(ans)



ans.to_csv('output.csv',index=False)