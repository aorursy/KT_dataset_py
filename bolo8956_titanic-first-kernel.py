# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd0=pd.read_csv(r"../input/train.csv",index_col=0,usecols=['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived'])

pd0.describe()



#pd0.groupby('Embarked')['Fare'].mean()
pd0=pd0.dropna()

pd0.describe()
pd0['Sex']=pd0['Sex'].replace({'male':0,'female':1})

pd0.groupby('Embarked')['Fare'].mean()
pd0['Embarked']=pd0['Embarked'].replace({'S':1,'Q':2,'C':3})
pd0.corr()
survived=pd0['Survived'].copy()

dftrain=pd0.drop(['Survived'],axis=1)



survived.head()

from sklearn.tree import DecisionTreeRegressor   

import numpy as np

rf=DecisionTreeRegressor()

rf.fit(dftrain,survived)



df_test=pd.read_csv(r"../input/test.csv",index_col=0,usecols=['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

df_test=df_test.fillna(method='pad')

df_test['Sex']=df_test['Sex'].replace({'male':0,'female':1})

df_test['Embarked']=df_test['Embarked'].replace({'S':1,'Q':2,'C':3})





rlt=rf.predict(df_test)



df_test.head()
def fun_conv(x):

    if x<=0.5:

        return 0

    else:

        return 1

rltdf=pd.DataFrame(rlt)



rltdf.columns=['Survived']

rltdf=rltdf.applymap(fun_conv)

rltdf

#comp1['show']=(comp1['pre']==comp1['act'])

#comp1[comp1['show']==True].count()
result=pd.DataFrame([df_test.index,rltdf['Survived'].values])

result=result.T



result.columns=['PassengerId','Survived']

result