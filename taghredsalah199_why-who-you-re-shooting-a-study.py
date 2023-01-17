import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_police= pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')

df_police
df_police.info()
plt.figure(figsize=(20,10))

sns.boxplot(x="race", y="age",hue="manner_of_death", data=df_police, palette="coolwarm")
plt.figure(figsize=(15,10))

sns.violinplot(x="gender", y="age",hue="signs_of_mental_illness", data=df_police,palette='rainbow')
df_police['race']=df_police['race'].astype('string')
plt.figure(figsize=(15,10))

sns.boxenplot(x="race", y="age",hue="threat_level", data=df_police)
plt.figure(figsize=(15,10))

sns.distplot(df_police['age'])

plt.figure(figsize=(15,10))

sns.boxenplot(x="flee", y="age", data=df_police,palette='Set1')
plt.figure(figsize=(15,10))



sns.stripplot(x="body_camera", y="age", data=df_police,hue='gender',jitter=True,palette='Set1')
plt.figure(figsize=(15,10))



sns.stripplot(x="threat_level", y="age", data=df_police,hue='gender',jitter=True)