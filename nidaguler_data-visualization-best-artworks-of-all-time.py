import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

import warnings

warnings.filterwarnings("ignore")

df=pd.read_csv("../input/best-artworks-of-all-time/artists.csv")
df.head()
df.tail()
df.drop(["id","bio","wikipedia"],axis=1,inplace=True)
df.head()
df_year = pd.DataFrame(df.years.str.split(' ',2).tolist(),columns = ['birth','-','death'])

df_year.drop(["-"],axis=1,inplace=True)

df["birth"]=df_year.birth

df["death"]=df_year.death

df.drop(["years"],axis=1,inplace=True)

df.info()
df["birth"]=df["birth"].apply(lambda x: int(x))

df["death"]=df["death"].apply(lambda x: int(x))
df["age"]=df.death-df.birth
df.age.describe()
df['age']=df['age']

bins=[30,55,65,77,98]

labels=["young adult","early adult","adult","senior"]

df['age_group']=pd.cut(df['age'],bins,labels=labels)

df.tail()
df = df.sort_values(by=["age"], ascending=False)

df['rank']=tuple(zip(df.age))

df['rank']=df.groupby('age',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values

df.drop(["rank"],axis=1,inplace=True)

df.reset_index(inplace=True,drop=True)

df.head()
df.sample(5)
df.describe()
df.isnull().sum()
df.nunique()
plt.figure(figsize=(18,5))



sns.barplot(x=df['nationality'].value_counts().index,y=df['nationality'].value_counts().values)

plt.title('nationality')

plt.xticks(rotation=75)

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()
plt.figure(figsize=(18,5))

sns.barplot(x=df['genre'].value_counts().index,

              y=df['genre'].value_counts().values)

plt.xlabel('genre')

plt.xticks(rotation=75)

plt.ylabel('Frequency')

plt.title('Show of genre Bar Plot')

plt.show()
plt.figure(figsize=(22,5))

sns.barplot(x = "genre", y = "paintings", hue = "age_group", data = df)

plt.xticks(rotation=75)

plt.show()
plt.figure(figsize=(17,8))

sns.barplot(x = "age_group", y = "paintings", hue = "genre", data = df)

plt.xticks(rotation=75)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
f,ax=plt.subplots(figsize=(9,10))

sns.barplot(x=df['age_group'].value_counts().values,y=df['age_group'].value_counts().index,alpha=0.5,color='red',label='age_group')

sns.barplot(x=df['genre'].value_counts().values,y=df['genre'].value_counts().index,color='blue',alpha=0.7,label='genre')

ax.legend(loc='upper right',frameon=True)

ax.set(xlabel='age_group , genre',ylabel='Groups',title="age_group vs genre ")

plt.show()
df['age'].unique()

len(df[(df['age']>50)].paintings)

f,ax1=plt.subplots(figsize=(25,10))

sns.pointplot(x=np.arange(1,41),y=df[(df['age']>50)].paintings,color='lime',alpha=0.8)



plt.xlabel('age>50 paintings')

plt.ylabel('Frequency')

plt.title('age>50 & paintings')

plt.xticks(rotation=90)

plt.grid()

plt.show()
plt.figure(figsize=(10,10))

sns.jointplot(x=np.arange(1,14),y=df[(df['age_group']=='young adult')].paintings,color='lime',alpha=0.8)

plt.xlabel('Young Adult index State')

plt.ylabel('Frequency')

plt.title('Young Adult Frequency Paintings')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(10,10))

sns.jointplot(x=np.arange(1,14),y=df[(df['age_group']=='senior')].paintings,color='lime',kind='hex',alpha=0.8)

plt.xlabel('Senior index State')

plt.ylabel('Frequency')

plt.title('Senior Frequency Paintings')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
df.age_group.unique()
labels=df['age_group'].value_counts().index

colors=['blue','red','yellow','green']

explode=[0.2,0.2,0.1,0.1,]

values=df['age_group'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Age Group According Analysis',color='black',fontsize=10)

plt.show()
plt.figure(figsize=(15,7))

sns.lmplot(x='age',y='paintings',data=df)

plt.xlabel('age')

plt.ylabel('paintings')

plt.title('age vs paintings')

plt.show()
sns.kdeplot(df['paintings'])

plt.xlabel('Values')

plt.ylabel('Frequency')

plt.title('paintings Kde Plot System Analysis')

plt.show()
sns.violinplot(df['paintings'])

plt.xlabel('paintings')

plt.ylabel('Frequency')

plt.title('Violin paintings Show')

plt.show()
sns.violinplot(x=df['age_group'],y=df['paintings'])

plt.show()
sns.violinplot(df['genre'][:10],df['paintings'][:10],hue=df['nationality'][:10],dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=90)

plt.show()
sns.set(style='whitegrid')

sns.boxplot(df['paintings'])

plt.show()
sns.boxplot(x=df['age_group'],y=df['paintings'])

plt.show()
plt.figure(figsize=(10,8))

sns.boxplot(x=df['nationality'][:10],y=df['paintings'][:10],hue=df['genre'][:10],palette="Set3")

plt.xticks(rotation=90)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()

sns.boxenplot(x="age_group", y="paintings",

              color="b",

              scale="linear", data=df)

plt.show()
sns.set(style='whitegrid')

sns.swarmplot(x=df['paintings'])

plt.show()
sns.swarmplot(x=df['genre'],y=df['paintings'])

plt.xticks(rotation=90)

plt.show()
sns.swarmplot(x=df['nationality'],y=df['paintings'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(df['age_group'])

plt.show()
ax = sns.distplot(df['age'])

plt.show()
df[df.age>85]
x=df[df.age>60].groupby('nationality')['paintings'].count().reset_index()

x
sns.countplot(df['nationality'])

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(df['age_group'],hue=df['genre'])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
sns.factorplot(x="genre", y="birth", hue="age_group", data=df,size=3, aspect=5)

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=90)

plt.show()
ax = sns.distplot(df['birth'])

plt.show()
sns.factorplot(x="genre", y="death", hue="age_group", data=df,size=3, aspect=5)

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=90)

plt.show()
ax = sns.distplot(df['death'])

plt.show()
sns.factorplot(x="genre", y="paintings", hue="age_group", data=df,size=3, aspect=5)

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xticks(rotation=90)

plt.show()
ax = sns.distplot(df['paintings'])

plt.show()
df[df['paintings']>750]