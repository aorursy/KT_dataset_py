import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("../input/uberdrives/My Uber Drives - 2016.csv")
df.isnull().sum()
df[df['END_DATE*'].isnull()]
df.drop(df.index[1155],inplace=True)
df.isnull().sum()
df.head()
df.dtypes
df['START_DATE*'] = df['START_DATE*'].astype('datetime64[ns]')

df['END_DATE*'] = df['END_DATE*'].astype('datetime64[ns]')
a=pd.crosstab(index=df['CATEGORY*'],columns='Count of travels as per category')

a.plot(kind='bar',color='r',alpha=0.7)

plt.legend()

a
plt.figure(figsize=(15,10))

sns.countplot(df['PURPOSE*'],order=df['PURPOSE*'].value_counts().index)
df.groupby('CATEGORY*')["MILES*"].mean().plot(kind='bar',color='g')

plt.axhline(df["MILES*"].mean(),label='Mean distance travelled per ride')

plt.legend()
df['Round_trip'] = df.apply(lambda x : 'Yes' if x['START*'] == x["STOP*"] else 'no',axis=1)
coun=pd.crosstab(df['Round_trip'],df['CATEGORY*'])

per=coun.div(coun.sum(1),axis=0)*100

per.plot(kind='bar',stacked=True)

plt.legend(bbox_to_anchor=(1.05,1),loc=2)

round(per,2)
df['Month'] = pd.DatetimeIndex(df['END_DATE*']).month

df['Month']
s= {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',

            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

df['Month'] = df['Month'].map(s)
df['Month'].dtypes
a=sns.countplot(df['Round_trip'],hue=df['Month'])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
c=pd.crosstab(index=df['Month'],columns='Number of trips')

c.sort_values('Number of trips',ascending=False).plot(kind='bar',color='y')

plt.axhline(c['Number of trips'].mean(),linestyle='--')
sns.countplot(df['Month'],hue=df['CATEGORY*'],order=df['Month'].value_counts().index)
df
df['Day/Nightride'] = pd.DatetimeIndex(df['START_DATE*']).time
a = pd.to_datetime(['18:00:00']).time
df['Day/Nightride'] = df.apply(lambda x : 'Night ride' if x['Day/Nightride'] > a else 'Day ride',axis=1)
sns.countplot(df['Day/Nightride'],hue=df['CATEGORY*'])
f = {}

for i in df['MILES*']:

    if i < 10:

        f.setdefault(i,'0-10 miles')

    elif i >= 10 and i < 20:

        f.setdefault(i,'10-20 miles')

    elif i >= 20 and i < 30:

        f.setdefault(i,'20-30 miles')

    elif i >= 30 and i < 40:

        f.setdefault(i,'30-40 miles')

    elif i >= 40 and i < 50:

        f.setdefault(i,'40-50 miles')

    else:

        f.setdefault(i,'Above 50 miles')
df['MILES*'] = df['MILES*'].map(f)
plt.figure(figsize=(10,10))

sns.countplot(df['MILES*'],order=df['MILES*'].value_counts().index)
f = pd.crosstab(df['Month'],df["MILES*"])

f.plot(kind='bar')

plt.legend(bbox_to_anchor=(1.05,1),loc=2)

f
z = df.groupby('Month')['Day/Nightride'].count().mean()
x,ax=plt.subplots(1,2,figsize=(10,10))

g = pd.crosstab(df['Month'],df["Day/Nightride"]).plot(kind='bar',ax=ax[0])

plt.axhline(z,color='g',linestyle='--',label='Mean number of travels across months')

sns.countplot(df['Month'],ax=ax[1])

plt.legend()
a=df.groupby('Month')['START*'].count()

b=a.index

plt.plot(b,df.groupby('Month')['START*'].count())