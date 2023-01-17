import numpy as np

import pandas as pd

import os
path="../input"
os.chdir(path)
df = pd.read_csv("crime.csv",  encoding = "ISO-8859-1")
df.info()
df.shape[0] # How many rows
df.shape[1] # How many columns
df.index.values # row names
df.columns.values # columns names
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot("SHOOTING", hue="DISTRICT", data = df) # Which district has reported more number of shootings
sns.countplot("SHOOTING", hue="YEAR", data = df) # Which year has reported more number of shootings
sns.countplot("SHOOTING", hue="DAY_OF_WEEK",data=df) # Which day of the week  has reported more number of shootings
sns.catplot(x="SHOOTING",       # Variable whose distribution (count) is of interest

            hue="DISTRICT",      # Show distribution, pos or -ve split-wise

            col="YEAR",       # Create two-charts/facets, gender-wise

            data=df,

            kind="count"

            )
plt.figure(figsize=(16,8))

df['DISTRICT'].value_counts().plot.bar()

plt.show()
# 2015

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2015].value_counts().plot.bar()

plt.show()



# 2016

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2016].value_counts().plot.bar()

plt.show()



# 2017

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2017].value_counts().plot.bar()

plt.show()



# 2018

plt.figure(figsize=(8,4))

df['DISTRICT'].loc[df['YEAR']==2018].value_counts().plot.bar()

plt.show()
data_shooting = df[df.SHOOTING == 'Y']

data_shooting.head()
df.head()
sns.countplot("OFFENSE_CODE_GROUP", hue="YEAR", data = df) 