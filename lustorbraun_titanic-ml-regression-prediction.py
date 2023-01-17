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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('/kaggle/input/machine-learning-on-titanic-data-set/train.csv')
df.head(2)
df.describe()
df.info()
df.isnull().sum()
df['Embarked'].fillna('C',inplace=True)
def age_reg(col):

    age=col[0]

    pclass=col[1]

    if pd.isnull(age):

        if pclass==1:

            age=np.mean(df[df['Pclass']==1]['Age'])//1

        elif pclass==2:

            age=np.mean(df[df['Pclass']==2]['Age'])//1

        else:

            age=np.mean(df[df['Pclass']==3]['Age'])//1

    return age

    

    

    



    

df['Age']=df[['Age','Pclass']].apply(age_reg,axis=1)
df.drop('Cabin',axis=1,inplace=True)
df.isnull().sum()
def emb(embarked):

    if embarked=='S':

        embarked=1

    elif embarked=='C':

        embarked=2

    elif embarked=='Q':

        embarked=3

    return embarked

df['Embarked']=df['Embarked'].apply(emb)
df['Sex']=pd.get_dummies(df['Sex'],drop_first=True)
df.columns
train=df
X_train=train.drop(['PassengerId','Name','Ticket','Survived'],axis=1)

y_train=train['Survived']
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
df2=pd.read_csv('/kaggle/input/machine-learning-on-titanic-data-set/test.csv')
df2.info()
df2.isnull().sum()
df2[df2['Fare'].isnull()]
df2.loc[152,'Fare']=np.mean(df2[df2['Pclass']==3]['Fare'])
def age_reg(col):

    age=col[0]

    pclass=col[1]

    if pd.isnull(age):

        if pclass==1:

            age=np.mean(df[df['Pclass']==1]['Age'])//1

        elif pclass==2:

            age=np.mean(df[df['Pclass']==2]['Age'])//1

        else:

            age=np.mean(df[df['Pclass']==3]['Age'])//1

    return age

    

    

    



    

df2['Age']=df2[['Age','Pclass']].apply(age_reg,axis=1)
df2.drop('Cabin',axis=1,inplace=True)
df2.head(2)
def emb(embarked):

    if embarked=='S':

        embarked=1

    elif embarked=='C':

        embarked=2

    elif embarked=='Q':

        embarked=3

    return embarked

df2['Embarked']=df['Embarked'].apply(emb)
df2['Sex']=pd.get_dummies(df2['Sex'],drop_first=True)
X_test=df2.drop(['PassengerId','Name','Ticket'],axis=1)
pred=model.predict(X_test)
df3=pd.read_csv('/kaggle/input/machine-learning-on-titanic-data-set/test.csv')
df3['Survived']=pred
df3.head()
df3.to_csv('results.csv')
df3[['PassengerId','Survived']].to_csv('results.csv')
df4=pd.read_csv('results.csv')

df4.head()
df4.head()
df4.drop('Unnamed: 0',axis=1,inplace=True)
df4.to_csv('results.csv',index=None)