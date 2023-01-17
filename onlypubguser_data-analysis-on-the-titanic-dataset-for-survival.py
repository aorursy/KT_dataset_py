# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/titanic/train.csv')

df
df2=df.drop(['Name','Ticket','Cabin'],axis='columns')

df2
def Correlation_with_survived(data):

    for i in data:

        print (df2[[i,'Survived']].groupby(i).describe())

    return 0
Correlation_with_survived(list(df2.columns[2:]))
FOB=df2['SibSp']+df2['Parch']

df2['FOB']=FOB

df3=df2.drop(['SibSp','Parch'],axis='columns')

df3
sns.factorplot(x='Sex',y='Age',hue="Survived",data=df3,kind='swarm')
sns.factorplot(x='Sex',y='Fare',hue="Survived",data=df3,kind='swarm')
sns.factorplot(x='Sex',y='Age',hue="Survived",data=df3,kind='swarm',col='Pclass')
sns.factorplot(x='Sex',y='Age',hue="Survived",data=df3,kind='swarm',col='Embarked')
sns.factorplot(x="FOB",data=df3,kind='count',col='Survived',hue='Sex')