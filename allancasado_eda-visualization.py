import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/toy-dataset/toy_dataset.csv")

del df["Number"]

df.head(20)
df.describe()
df.info()
df.shape
df.columns
df.dtypes
df.City.unique()
df.Gender.unique()
df.Illness.unique()
df.isnull().sum()
is_error = df[df["Income"] < 0]

is_error
df = df.drop(df.index[245])

df.describe()
plt.figure(figsize=(10,5))

sns.countplot(x="Illness", palette="rocket", data=df)
plt.figure(figsize=(10,5))

sns.countplot(x="Gender", palette="rocket", data=df)
plt.figure(figsize=(10,5))

sns.countplot(y="City", palette="rocket", data=df)
plt.figure(figsize=(10,5))

sns.countplot(x="Gender", hue="Illness",palette="rocket", data=df)
plt.figure(figsize=(10,5))

sns.countplot(x="Gender", hue="City",palette="rocket", data=df)
plt.figure(figsize=(10,5))

sns.countplot(x="City", hue="Gender",palette="rocket", data=df)
#colors: 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
plt.figure(figsize=(10,5))

sns.distplot(df["Age"], color='b')

plt.title("Age distribution")
plt.figure(figsize=(10,5))

sns.distplot(df["Income"], color='g')

plt.title("Income distribution")
plt.figure(figsize=(10,5))

sns.distplot(df[df["Illness"] == "Yes"]["Income"], color="Y")

plt.title("Income distribution - Illness:Yes")
plt.figure(figsize=(10,5))

sns.distplot(df[df["Illness"] == "No"]["Income"], color="m")

plt.title("Income distribution - Illness:No")
fig = plt.figure(figsize=(10,5))

sns.distplot(df[df["Gender"] == "Male"]["Income"])

sns.distplot(df[df["Gender"] == "Female"]["Income"])

fig.legend(labels=['Male','Female'])

plt.title("Income distribution - Man and Woman")

plt.show()
plt.figure(figsize=(10,5))

sns.boxplot(x=df["Income"], y=df["City"], data=df)