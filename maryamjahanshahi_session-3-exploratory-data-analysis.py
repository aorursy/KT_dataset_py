import pandas as pd

import numpy as np

import matplotlib
df = pd.read_csv("../input/salary-dataset/salary_dataset.csv")
df.head(25)
df["jobTitle"].describe()
df['jobTitle'].unique()
df['jobTitle'].value_counts()
df['jobTitle'].value_counts().plot(kind='bar')
df['age'].describe()
df['age'].hist(bins=5)
df['gender'].value_counts().plot(kind='bar')
df['edu'].value_counts().plot(kind='bar')
df['dept'].value_counts().plot(kind='bar')
df.shape
df.groupby(['jobTitle', 'gender']).median()
df['totalComp'] = df['basePay']+df['bonus']
df['bonusPct'] = round(df['bonus']*100/df['basePay'])
df.head()