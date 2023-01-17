import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('../input/titanic/train.csv')

data.info()
data.isnull().sum()
df_total=data.groupby('Pclass')['Survived'].count()

df_total
df_death=data[data.Survived == 0].groupby('Pclass')['Survived'].count()

df_death
df_acc=df_death / df_total

df_acc
df_acc.plot(kind='bar')
df_acc=df_acc.reset_index()

df_acc.columns=['Pclass',"Survived"]

r=(df_acc.Pclass).corr(df_acc.Survived)

print(str(r))