# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
sns.set_style("darkgrid")
%matplotlib inline
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.describe()
df.info()
#DATA CLEANING
print('Mean age of pclass 1 is {}, 2 is {} and 3 is {}'.format(math.ceil(df[df['Pclass']==1]['Age'].mean()),math.ceil(df[df['Pclass']==2]['Age'].mean()),math.ceil(df[df['Pclass']==3]['Age'].mean())))    
def fnc(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
            
        if pclass ==1:
            return 39
        elif pclass ==2:
            return 30
        else:
            return 26
    else:
        return age
df['Age']=df[['Age','Pclass']].apply(fnc,axis=1)
df.drop('Cabin',axis=1,inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#DATA VISUALISATION
sns.heatmap(df[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True,cmap='coolwarm')
sns.catplot(x='SibSp',y='Survived',data=df,kind='bar')
sns.countplot(x='Survived',hue='Sex',data=df,palette='Set3')
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=df,kind="bar", palette="RdBu")
sns.scatterplot(x='Age',hue='Sex',y='Fare',data=df)
sns.countplot(x='Embarked',data=df)
sns.distplot(df['Age'],kde=False,bins=30)
sns.relplot(x="Age", y="Fare",col='Sex',kind="scatter", data=df)
sns.boxplot(x='Pclass',y='Age',data=df,palette='RdBu')
sns.distplot(df['Fare'],color='Blue')
#This is my first project 
#Thank You