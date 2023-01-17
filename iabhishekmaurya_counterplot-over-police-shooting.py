import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')
data.head()
data.info()
data.isnull().sum()
data['manner_of_death'].value_counts()
print("The average person age {:.4f} players, 99% of people have {} age or less, while the highest age is {} and the lowest is {}".format(data['age'].mean(),data['age'].quantile(0.99), data['age'].max(), data['age'].min()))
data.groupby(['gender','manner_of_death'])['manner_of_death'].count()
data['manner_of_death'].replace({'shot' : 0,'shot and Tasered' : 1},inplace=True)

data
data['date'].describe()
data['state'].value_counts()
ax = sns.countplot(x="state", data=data)
ax = sns.countplot(x="flee", data=data)
ax = sns.countplot(x="signs_of_mental_illness", data=data)
ax = sns.countplot(x="gender", data=data)
ax = sns.countplot(x="race", data=data)