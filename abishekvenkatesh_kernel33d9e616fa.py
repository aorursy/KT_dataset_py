# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing,CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(15)


df.drop(["Ticket","Name","Fare",], axis = 1, inplace = True) 

df.head(10)


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='SibSp', data=df, palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Embarked', data=df, palette='RdBu_r')
df['Embarked'].hist(color='green',bins=40,figsize=(8,4))
df['Age'].hist(color='green',bins=40,figsize=(8,4))
df.head(10)
sns.kdeplot(df["Age"])
#df["Age"] = df["Age"].apply(lambda x: math.log(x))
sns.kdeplot(df["Age"])

sns.boxplot(df["Age"])
df.shape


sns.boxplot(df["Age"])

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df["Age"]))
print(z)
ab=[]
ab.append(np.where(z>3))

print(ab)
for i in ab:
    print(i)
df=df.drop(df.index[ab])

df.shape

plt.figure(figsize=(14, 7))
sns.boxplot(x='Embarked',y='Age',data=df,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Embarked = cols[1]
    
    if pd.isnull(Age):
        if Embarked == 'S':
            return 27
        elif Embarked == 'C':
            return 29
        else:
            return 25
    else:
         return Age
    
    
df['Age'] = df[['Age','Embarked']].apply(impute_age,axis=1)
df.head(5)
df.drop(["Cabin"], axis = 1, inplace = True)
df.head()

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Sex'] = df['Sex'].cat.codes
df['Embarked'] = df['Embarked'].cat.codes
df.head()
dfp=pd.read_csv('/kaggle/input/titanic/test.csv')
dfp.head()
dfp.drop(["Ticket","Name","Fare",], axis = 1, inplace = True) 
dfp.head()
dfp.isnull().sum()
dfp.head()
plt.figure(figsize=(14, 7))
sns.boxplot(x='Embarked',y='Age',data=dfp,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Embarked = cols[1]
    
    if pd.isnull(Age):
        if Embarked == 'S':
            return 24
        elif Embarked == 'C':
            return 35
        else:
            return 25
    else:
         return Age
    
    
dfp['Age'] = dfp[['Age','Embarked']].apply(impute_age,axis=1)
df.head()
x=df.iloc[:,2:8].values
y=df.iloc[:,1].values
print(y)
from sklearn.linear_model import LogisticRegression
#create an instance and fit the model 
logmodel = LogisticRegression()
logmodel.fit(x, y)
dfp['Sex'] = dfp['Sex'].astype('category')



dfp['Embarked'] = dfp['Embarked'].astype('category')
dfp['Sex']=dfp['Sex'].cat.codes
dfp["Embarked"]=dfp["Embarked"].cat.codes
dfp.head()
dfp=dfp.drop("Cabin",axis=1)
dfp.head()
x_test=dfp.iloc[:,1:7].values
print(x_test)
dfp.head()
Predictions = logmodel.predict(x_test)

print(Predictions)

#print(len(Predictions))
pred=pd.DataFrame(Predictions,columns=["Survived"])


pred["PassengerId"]=dfp["PassengerId"]
pred["Survived"]=pred["Survived"]
pred.head()

predf=pd.DataFrame(dfp["PassengerId"],columns=["PassengerId"])
predf.head()
predf["Survived"]=pred["Survived"]
predf.head()
predf.to_csv (r'export_dataframe2.csv', index = False, header=True)