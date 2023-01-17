# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt #导入





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_sub=pd.read_csv('../input/gender_submission.csv')



# Any results you write to the current directory are saved as output.
df_train.head()
df_train=df_train.drop(['PassengerId','Ticket','Name','Embarked'],axis=1) 

df_test=df_test.drop(['PassengerId','Ticket','Name','Embarked'],axis=1) 

df_test.mean()
df_test=df_test.fillna(df_test.mean())

#df_test

df_train=df_train.fillna(df_train.mean())

#df_train
m_train=df_train.values

m_test=df_test.values



m_train[m_train[:,2]=='male',2]=1

m_train[m_train[:,2]=='female',2]=0





m_test[m_test[:,1]=='male',1]=1

m_test[m_test[:,1]=='female',1]=0
cabins=list(set(m_train[:,7]))

cabinMap={}

for i in range(len(cabins)):

    cabinMap[cabins[i]]=i

    

for i in range(m_train.shape[0]):

    m_train[i,7]=cabinMap[m_train[i,7]]



for i in range(m_test.shape[0]):

    if m_test[i,6] in cabinMap:

        m_test[i,6]=cabinMap[m_test[i,6]]

    else:

        cabinMap[m_test[i,6]]=len(cabinMap)

        m_test[i,6]=cabinMap[m_test[i,6]]




x=m_train[:,1:]

y=m_train[:,0]



xt=m_test



y=y.astype('int')
test = pd.read_csv('../input/test.csv')

test['Survived'] = 0

test.loc[test['Sex'] == 'female','Survived'] = 1

data_to_submit = pd.DataFrame({

    'PassengerId':test['PassengerId'],

    'Survived':test['Survived']

})

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x,y)



yt=logreg.predict(xt)

test = pd.read_csv('../input/test.csv')

test['Survived'] = yt

data_to_submit = pd.DataFrame({

    'PassengerId':test['PassengerId'],

    'Survived':test['Survived']

})
data_to_submit
data_to_submit.to_csv('csv_to_submit.csv', index = False)