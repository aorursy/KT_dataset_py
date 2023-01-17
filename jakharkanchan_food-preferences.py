# import the libraries

import pandas as pd

import numpy  as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/food-preferences/Food_Preference.csv')

print('Shape of Original Dataset is : {}'.format(df.shape))
df.sample(5)
#As we are not going to use Timestamp and Participant_ID, so we can drop these columns

df = df.drop(['Timestamp','Participant_ID'],axis=1)

df.columns
df.dtypes
((df.isna()).sum()/len(df))*100
df = df.dropna()
print('Shape of Cleaned Dataset is : {}'.format(df.shape))
df.head()
# Lets look at distribution of Juice and Dessert Variables

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x=df['Juice'],ax=ax1)

sns.countplot(x=df['Dessert'],ax=ax2)
# Lets look at distribution of Gender and Food Variables

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x=df['Gender'],ax=ax1)

sns.countplot(x=df['Food'],ax=ax2)

# Lets look at distribution of Nationality Variable

fig = plt.figure(figsize=(10,20))

ax1 = fig.add_subplot(1,1,1)

sns.countplot(y=df['Nationality'],ax=ax1)
# Lets look at distribution of Age Variable

fig = plt.figure(figsize=(10,20))

ax1 = fig.add_subplot(1,1,1)

sns.countplot(y=df['Age'],ax=ax1)
# Age v/s Dessert

fig = plt.figure(figsize=(10,3))

ax1 = fig.add_subplot(1,1,1)

sns.boxplot(x=df['Age'],y=df['Dessert'])
# Age vs Food

fig = plt.figure(figsize=(10,3))

ax1 = fig.add_subplot(1,1,1)

sns.boxplot(x=df['Age'],y=df['Food'])
# Age v/s Juice

fig = plt.figure(figsize=(10,3))

ax1 = fig.add_subplot(1,1,1)

sns.boxplot(x=df['Age'],y=df['Juice'])