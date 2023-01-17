import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
df = pd.read_csv("../input/crimes-in-boston/crime.csv",encoding = "ISO-8859-1")

df.isnull().sum()
del df['SHOOTING']
df.describe()
df.head(10)
df[df.columns[df.isnull().any()]].head(20)

df.head(20)
df.dropna(inplace=True)

df.isnull().sum()
df.shape
##for x,y in zip(df["OFFENSE_CODE_GROUP"].value_counts().index,df["OFFENSE_CODE_GROUP"].value_counts()):

 #     print(x,":",y)

df["OFFENSE_CODE_GROUP"].value_counts().plot(kind="bar", figsize=(18, 10))
df["OFFENSE_CODE_GROUP"].value_counts().plot(kind="pie", figsize=(22, 12), autopct='%1.1f%%')
df["DISTRICT"].value_counts().plot(kind="pie", figsize=(22, 12), autopct='%1.1f%%')
cat = [df["OFFENSE_CODE_GROUP"].value_counts().index[x] for x in range(9)]

fig, ax = plt.subplots(3, 3, figsize=(20, 10))

for var, subplot in zip(cat, ax.flatten()):

    df[df["OFFENSE_CODE_GROUP"]==var]["DISTRICT"].value_counts().plot(kind="bar",ax=subplot)

    subplot.set_ylabel(var)

fig.subplots_adjust(left=0.2, wspace=0.4, hspace = 0.6)
cat1 = [df["DISTRICT"].value_counts().index[x] for x in range(12)]

r = pd.DataFrame()

r["OFFENSE_CODE_GROUP"],r["DISTRICT"],cat =  df["OFFENSE_CODE_GROUP"],df["DISTRICT"],[df["OFFENSE_CODE_GROUP"].value_counts().index[x] for x in range(9,63)]

for x in cat:

    r.drop(r[r["OFFENSE_CODE_GROUP"]==x].index,inplace=True)

cat11,cat12 = cat1[:len(cat1)//2],cat1[len(cat1)//2:]



fig, ax = plt.subplots(2, 3, figsize=(20, 10))

for var, subplot in zip(cat11, ax.flatten()):

    r[r["DISTRICT"]==var]["OFFENSE_CODE_GROUP"].value_counts().plot(kind="bar",ax=subplot)

    subplot.set_ylabel(var)

fig.subplots_adjust(left=0.2, wspace=0.4, hspace = 0.6)
for var, subplot in zip(cat11, ax.flatten()):

    r[r["DISTRICT"]==var]["OFFENSE_CODE_GROUP"].value_counts().plot(kind="bar",ax=subplot)

    subplot.set_ylabel(var)

fig.subplots_adjust(left=0.2, wspace=0.4, hspace = 0.6)
r["OCCURRED_ON_DATE"] = df["OCCURRED_ON_DATE"]

r["OCCURRED_ON_DATE"]=[x[:10] for x in r["OCCURRED_ON_DATE"]]

r["OCCURRED_ON_DATE"]=pd.to_datetime(pd.Series(r["OCCURRED_ON_DATE"]))

r["count"] = [1 for x in r["OCCURRED_ON_DATE"]]

res = r.pivot_table(index="OCCURRED_ON_DATE",columns="OFFENSE_CODE_GROUP",values="count",aggfunc='sum')
res.dropna(inplace=True)

res = pd.DataFrame(res.to_records())
f = res.set_index('OCCURRED_ON_DATE')

f.index
cat3 = f.columns

cat31,cat32,cat33 = cat3[:3],cat3[3:6],cat3[6:]
for x in cat31:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 10))

     plt.legend(loc = x)

plt.show()
for x in cat32:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 10))

     plt.legend(loc = x)

plt.show()
for x in cat33:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 10))

     plt.legend(loc = x)

plt.show()
for x in cat3:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 10))

     plt.legend(loc = x)

plt.show()
res1 = r.pivot_table(index="OCCURRED_ON_DATE",columns="DISTRICT",values="count",aggfunc='sum')

res1.dropna(inplace=True)

res1 = pd.DataFrame(res1.to_records())

f = res1.set_index('OCCURRED_ON_DATE')

f.index

cat3 = f.columns

cat31,cat32,cat33,cat34 = cat3[:3],cat3[3:6],cat3[6:9],cat3[9:12]
for x in cat31:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 6))

     plt.legend(loc = x)

plt.show()
for x in cat32:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 6))

     plt.legend(loc = x)

plt.show()
for x in cat33:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 6))

     plt.legend(loc = x)

plt.show()
for x in cat34:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 6))

     plt.legend(loc = x)

plt.show()
for x in cat3:

     y = f[x].resample('MS').mean()

     y.plot(figsize=(15, 6))

     plt.legend(loc = x)

plt.show()