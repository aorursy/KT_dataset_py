import matplotlib.pyplot as plt
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
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Survived'].value_counts()
pd.crosstab(df['Sex'],df['Survived'],normalize='index')
pd.crosstab(df['Pclass'],df['Survived'],normalize='index')
pd.crosstab(df['Embarked'],df['Survived'],normalize='index')
pd.crosstab(df['SibSp'],df['Survived'],normalize='index')
df['SibSp'].value_counts()
pd.crosstab(df['Parch'],df['Survived'],normalize='index')
def func(d,s):

    d.apply(lambda x:'green'if x[s]==0 else 'red',axis=1)

    
plt.scatter(df['Age'],df['Fare'],c=df.apply(lambda x:'red'if x['Survived']==0 else 'green',axis=1))

plt.xlabel('Age')

plt.ylabel('Fare')
df['Fare'].plot.box()
np.log(df['Fare']+5).plot.hist()
df['Age'].plot.hist()
df['Fare']=np.log(df['Fare']+5)
df['Fare'].plot.hist(bins=20)

plt.xlabel('Fare')
df.isnull().sum()
df1=pd.read_csv('/kaggle/input/kernel9650ec13eb/cleaned.csv')
df1.drop(['Fare'],axis=1,inplace=True)
df1['Fare']=df['Fare']
df1.to_csv('cleaned1.csv')