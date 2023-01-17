import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

df = pd.read_csv('../input/adhd_hila/Users For Hila.csv')

df.dropna(subset=['ADHD','Driving Experience', 'Age', 'Gender'], inplace=True)



positive = 'יש'

negative = 'אין'

df.ADHD = df.ADHD.map({ positive : 1, negative : 0})

df.Gender = df.Gender.map({ 'Female' : 'F', 'Male' : 'M'})

df.rename(index=str, columns={"ADHD": "ADHD_orig"}, inplace=True)

df.rename(index=str, columns={"Gender": "Gender_user"}, inplace=True)

df.set_index("UID PM", inplace=True)

df.drop(['Unnamed: 0'] ,axis=1, inplace=True)

print(df.shape)

df.head()

ax = df['Driving Experience'].hist()

ax.set_xlabel("Driving Experience", labelpad=20, weight='bold', size=12)

ax.set_ylabel("Count", labelpad=20, weight='bold', size=12)

plt.show()



plt.hist(df[df.ADHD_orig == 0]['Driving Experience'], alpha=0.5, label='Without ADHD')

plt.hist(df[df.ADHD_orig == 1]['Driving Experience'], alpha=0.5, label='With ADHD')

plt.legend(loc='upper right')

plt.show()
tmp = df[df['Driving Experience'] < 15]

plt.hist(tmp[tmp.ADHD_orig == 0]['Driving Experience'], alpha=0.5, label='Without ADHD')

plt.hist(tmp[tmp.ADHD_orig == 1]['Driving Experience'], alpha=0.5, label='With ADHD')

plt.legend(loc='upper right')

plt.show()
ax = df['Age'].hist()

ax.set_xlabel("Age", labelpad=20, weight='bold', size=12)

ax.set_ylabel("Count", labelpad=20, weight='bold', size=12)
print("AGE 7 and 11 doesn't make sense")

df[df.Age < 15].head()
print("Remove AGE < 15")

df = df[df.Age >= 15]
df.ADHD_orig.value_counts().plot(kind='pie')
df.Gender_user.value_counts().plot(kind='pie')
df_data = pd.read_csv('../input/hila-data/largedata.csv', encoding ='Windows-1255')

df_data.reset_index()

df_data.set_index("UID.PM", inplace=True)

df_data.ADHD = df_data.ADHD.map({ 'Yes' : 1, 'No' : 0})

df_data.shape
df_data
result = df_data.join(df, how='outer')

result.dropna(subset=['Event','Age'], inplace=True)

print(result.shape)

result
print(len(result[result.Gender != result.Gender_user] ))

print(result[result.ADHD != result.ADHD_orig].index.value_counts())

result.drop([5528, 5557], inplace=True)

result.shape

corrmat = result[["ADHD","Driving Experience","Age","Gender"]].corr(method='spearman')

f, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)