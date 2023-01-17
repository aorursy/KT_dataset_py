import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/data-police-shootings/fatal-police-shootings-data.csv")
df.head()
df.replace(to_replace = ['W'], value = ['White and Non-Hispanic'], inplace = True)

df.replace(to_replace = ['B'], value = ['Black and Non-Hispanic'], inplace = True)

df.replace(to_replace = ['H'], value = ['Hispanic'], inplace = True)

df.replace(to_replace = ['N'], value = ['Native American'], inplace = True)

df.replace(to_replace = ['O'], value = ['Other'], inplace = True)

df.replace(to_replace = ['A'], value = ['Asian'], inplace = True)
df['date']=pd.to_datetime(df['date'])

df['year']=pd.to_datetime(df['date']).dt.year

df['month']=pd.to_datetime(df['date']).dt.month
df.head()
df['armed'].value_counts()
df['threat_level'].value_counts()
df['flee'].value_counts()
df['signs_of_mental_illness'].value_counts()
df["age"].plot.hist()
sns.countplot(x = "manner_of_death", data = df)
sns.set(rc={'figure.figsize':(15,5)})

sns.countplot(x = "race", data = df)


sns.countplot(x = "state", data = df)
sns.countplot(x = "signs_of_mental_illness", data = df)
sns.countplot(x = "flee", data = df)
sns.countplot(x = "body_camera", data = df)
sns.countplot(x = "year", data = df)
sns.countplot(x = "race", hue = "flee",data = df, palette = 'Set2')
sns.countplot(x = "race", hue = "body_camera",data = df, palette = 'Set2')
sns.countplot(x = "state", hue = "flee",data = df, palette = 'Set2')