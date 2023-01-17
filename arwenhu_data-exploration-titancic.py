# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
data.head()
data.describe()
data.dtypes
data.apply(lambda x:sum(x.isnull()),axis=0)
data['Sex'].value_counts()
def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(data['Survived'],data['Pclass'],margins=True).apply(percConvert,axis=0)
pd.crosstab(data['Survived'],data['Sex'],margins=True).apply(percConvert,axis=0)
data[['Age','Sex','Pclass']].groupby(['Sex','Pclass']).mean()
data['Embarked'].value_counts()
data.groupby('Embarked').describe()
data.loc[data['Embarked'].isnull()]
data['Embarked'].fillna('S', inplace=True)
data['Embarked'].value_counts()
data[['Fare','Pclass']].groupby('Pclass').boxplot(column='Fare')
def getSalution(string):
    salstr = string.split(',')[1].split('.')[0].strip()
    return salstr if len(salstr) > 0 else 'NULL'
data['Salution'] = data['Name'].apply(getSalution)
data['Salution'].value_counts()
def groupSalution(string):
    if string == 'Mr' or string == 'Miss' or string == 'Master' or string == 'Mrs':
        return string
    else:
        return 'Others'
data['groupSalution'] = data['Salution'].apply(groupSalution)
data['groupSalution'].value_counts()
g = sns.swarmplot(x="groupSalution", y="Age", hue="Pclass",dodge=True, data=data)
plt.xticks(rotation = 45)
impute_grps = data.pivot_table(values='Age', index=['groupSalution'], columns=['Pclass', 'Sex'], aggfunc=np.median)
print(impute_grps)
def fage(x):
    return impute_grps[x['Pclass']][x['Sex']][x['groupSalution']]
# Replace missing values 
data['Age'].fillna(data[data['Age'].isnull()].apply(fage, axis=1), inplace=True)
#data.fillna(data[data['Age'].isnull()].apply(fage,axis=1), inplace=True)
sns.swarmplot(x="groupSalution", y="Age", hue="Pclass",dodge=True, data=data)
data[data['Age'].isnull()]