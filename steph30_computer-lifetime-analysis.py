import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/computerlifetime/computers.csv')

df
df.head()
df.info()
df.team.unique()

df.provider.unique()
df.describe()
# Check Duplicate enteries

df.duplicated().sum()



# No dupliate Entries
# Check fr missing values

df.isnull().sum()
df.skew()
#Filling up null values



df.temperatureInd.fillna(df.temperatureInd.mean(),inplace=True)

df.pressureInd.fillna(df.pressureInd.median(),inplace=True)

df.isnull().sum()

plt.figure(figsize=(18,10))

plt.scatter(df.index,df.moistureInd)

plt.show()



# the dot which is more than 1000 is outlier
#drop the row which contains outlier

df=df[df.moistureInd<1000]

df.shape
df.skew()

# skew should always be bw 1 and -1
df.columns
cats = ['broken','team','provider']

for col in cats:

    plt.figure(figsize=(8,4))

    sns.countplot(df[col])

    plt.show()
numeric = ['lifetime','pressureInd','moistureInd','temperatureInd']

for col in numeric:

    plt.figure(figsize=(8,4))

    plt.hist(df[col])

    plt.title(col)

    plt.show()
# lifetime vs broken

plt.figure(figsize=(12,5))

sns.distplot(df.lifetime[df.broken==0])

sns.distplot(df.lifetime[df.broken==1])

plt.legend(['0 - not broken','1 - broken'])

plt.show()
plt.figure(figsize=(12,5))

sns.distplot(df.temperatureInd[df.broken==0])

sns.distplot(df.temperatureInd[df.broken==1])

plt.legend(['0 - not broken','1 - broken'])

plt.show()
# pressure vs broken

plt.figure(figsize=(12,5))

sns.distplot(df.pressureInd[df.broken==0])

sns.distplot(df.pressureInd[df.broken==1])

plt.legend(['0 - not broken','1 - broken'])

plt.show()
# team vs broken

plt.figure(figsize=(6,3))

sns.countplot(df['team'],order=df.team.unique())

plt.show()



plt.figure(figsize=(6,3))

sns.countplot(df['team'][df.broken==1],order=df.team.unique())

plt.show()
# team vs broken

out=pd.crosstab(df.team,df.broken,margins=True)

out
(out[1]/out['All']) * 100
# provider vs broken

out=pd.crosstab(df.provider,df.broken,margins=True)

out
(out[1]/out['All']) * 100
# lifetime vs team vs broken

plt.figure(figsize=(12,5))

sns.swarmplot(x='team',y='lifetime',hue='broken',data=df)

plt.show()
# lifetime vs provider vs broken

plt.figure(figsize=(12,5))

sns.swarmplot(x='provider',y='lifetime',hue='broken',data=df)

plt.show()
# lifetime vs moisture vs broken

plt.figure(figsize=(12,5))

sns.swarmplot(x='lifetime',y='moistureInd',hue='broken',data=df)

plt.show()
sns.pairplot(df)

plt.show()