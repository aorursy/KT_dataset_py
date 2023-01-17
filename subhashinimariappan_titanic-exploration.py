import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.head()
df_train.info()
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df_train.isnull(),cmap='YlGnBu')

df_train.Cabin.isnull().value_counts()
df_train=df_train.drop('Cabin',axis=1)
df_train.head()
df_train.Age.isnull().value_counts()
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train)
df.groupby('Pclass').mean().Age
def age(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
        if pclass==1:
            return 38
        elif pclass==2:
            return 29
        else:
            return 25
    else:
        return age
df_train['Age'] = df_train[['Age','Pclass']].apply(age,axis=1)
sns.heatmap(df_train.isnull(),cmap='YlGnBu')
sns.countplot('Survived',data=df_train)
sns.countplot('Survived',hue='Sex',data=df_train)
sns.countplot('Pclass',hue='Survived',data=df_train)
sns.set_style('whitegrid')
sns.distplot(df_train['Age'],kde=False,bins=30)
sns.countplot(x='SibSp',data=df_train)

df_train['Fare'].hist(bins=40,figsize=(8,4))





