import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('../input/police-officer-deaths-in-the-us/clean_data.csv',parse_dates=['date'])

df.head()
df.shape
df.isnull()
sns.set_style('whitegrid')

sns.set_context('poster')
plt.figure(figsize=(8, 4))

df.canine.value_counts().plot(kind='bar',color='m')

plt.title("Overview of Death")

plt.ylabel("Deaths")

plt.xticks(df.canine.value_counts().index,['Police', 'Dogs'],rotation=0)

plt.show()
df.canine.value_counts()
ydf=df[df['canine']==False].year.value_counts().sort_index()

plt.figure(figsize=(15,10))

sns.lineplot(ydf.index,ydf.values)

plt.title('police officers deaths since year 1791')

plt.xlabel('Years')

plt.ylabel('Deaths')

plt.show()
yddf=df[df['canine']==True].year.value_counts().sort_index()

plt.figure(figsize=(10,8))

sns.lineplot(yddf.index,yddf.values)

plt.title('Dogs death since 1791')

plt.xlabel('Years')

plt.ylabel('Deaths')

plt.show()
yddf=df[df['canine']==True].cause_short.value_counts()

plt.figure(figsize=(10,8))

sns.barplot(yddf.index,yddf.values)

plt.title('Dogs death by Cause')

plt.xlabel('Cause')

plt.ylabel('Deaths')

plt.xticks(rotation=90)

plt.show()
yddf=df[df['canine']==True].state.value_counts()

plt.figure(figsize=(10,8))

sns.barplot(yddf.index,yddf.values,palette='coolwarm')

plt.title('Dogs death by State')

plt.xlabel('States')

plt.ylabel('Deaths')

plt.xticks(rotation=90)

plt.show()
#DataFrame

#ydf=df[(df['canine']==True)].groupby(['year']).cause.count().reset_index()

#plt.figure(figsize=(10,8))

#sns.lineplot(x='year',y='cause',data=ydf)

#plt.title('Dogs death since 1791')

#plt.xlabel('Years')

#plt.ylabel('Deaths')

#plt.show()
sns.set_context('talk')

cdf=df.cause_short.value_counts()

plt.figure(figsize=(12,10))

sns.barplot(cdf.index,cdf.values)

plt.title('Deaths by cause')

plt.xlabel('Deaths')

plt.ylabel('Cause')

plt.xticks(rotation=90)

plt.show()
sdf=df['state'].value_counts()

plt.figure(figsize=(30,10))

sns.barplot(sdf.index,sdf.values,palette='ocean')

plt.title('Deaths by states')

plt.xlabel('STATES')

plt.ylabel('Deaths')

plt.xticks(rotation=90)

plt.show()
ssdf=df['state'].value_counts()

sdf=ssdf.head(10)

plt.figure(figsize=(30,10))

sns.barplot(sdf.index,sdf.values,palette='Paired')

plt.title('Top 10 states in deaths')

plt.xlabel('STATES')

plt.ylabel('Deaths')

plt.show()
sns.set_context('talk')

plt.figure(figsize=(15, 8))

texs=df[(df['state']==" TX")].cause_short.value_counts()

sns.barplot(texs.index,texs.values,palette='rocket')

plt.xticks(rotation=90)

plt.ylabel("Deaths")

plt.xlabel('Cause')

plt.title('Texas Deaths by cause')

plt.show()
sns.set_context('talk')

fig=plt.figure(figsize=(15, 8))

df[(df['state']==" TX")].year.value_counts().sort_index().plot()

plt.ylabel("Deaths")

plt.xlabel('Years')

plt.title('Texas Deaths over Years')

plt.show()

fig.savefig('poy.jpg')
sns.set_context('talk')

plt.figure(figsize=(15,8))

ddf=df['dept_name'].value_counts().reset_index()

d=ddf.head(5)

sns.barplot(x='index',y='dept_name',data=d)

plt.xticks(rotation=70)

plt.xlabel('Department')

plt.ylabel('Deaths')

plt.title('Top 5 Police officer deaths by department')

plt.show()
df['cause_short'].unique()
r=df[df['cause_short'].isin(["Automobile accident","Bicycle accident","Motorcycle accident","Struck by streetcar","Struck by vehicle","Vehicle pursuit","Vehicular assault"])]

plt.figure(figsize=(15,8))

r.cause_short.value_counts().plot(kind='bar',color='lightseagreen')

plt.xticks(rotation=70)

plt.xlabel('Cause of Deaths by Road Accidents')

plt.ylabel('Deaths')

plt.title('Road Accidents deaths')

plt.show()
sns.set_context('talk')

plt.figure(figsize=(15,8))

rss=r.state.value_counts()

sns.barplot(rss.index,rss.values,palette='gist_rainbow')

plt.xticks(rotation=70)

plt.xlabel('States')

plt.ylabel('Deaths')

plt.title('Police road accident deaths by states')

plt.show()
plt.figure(figsize=(15,8))

r.dept_name.value_counts().plot()

plt.xticks(rotation=50)

plt.xlabel('Departments')

plt.ylabel('Deaths')

plt.title('Police road accident deaths by Departments')

plt.show()
plt.figure(figsize=(15,8))

r.year.value_counts().sort_index().plot(color='m')

plt.xlabel('Years')

plt.ylabel('Deaths')

plt.title('Police road accident deaths by Year')

plt.show()
med=df[df['cause_short'].isin(['Heart attack', 'Duty related illness', 'Heat exhaustion','Asphyxiation'])]

plt.figure(figsize=(15,8))

ms=med.cause_short.value_counts()

sns.barplot(ms.index,ms.values,palette='Set1')

plt.xticks(rotation=0)

plt.xlabel('Cause of deaths by Medical Related')

plt.ylabel('Deaths')

plt.title('Medical Related Deaths')

plt.show()
sns.set_context('talk')

plt.figure(figsize=(15,8))

mss=med.state.value_counts()#.plot(kind='bar',color='slateblue')

sns.barplot(mss.index,mss.values,palette='cubehelix')

plt.xticks(rotation=80)

plt.xlabel('States')

plt.ylabel('Deaths')

plt.title('medical deaths by states')

plt.show()
fig=plt.figure(figsize=(15,8))

med.year.value_counts().sort_index().plot(color='deepskyblue')

plt.xlabel('Years')

plt.ylabel('Deaths')

plt.title('Medical related deaths by Year')

plt.show()

fig.savefig('ppy.jpg')
plt.figure(figsize=(15,8))

med.dept_name.value_counts().plot()

plt.xticks(rotation=70)

plt.xlabel('Departments')

plt.ylabel('Deaths')

plt.title('Medical deaths by Departments')

plt.show()
ter=df[df['cause_short'].isin(['Terrorist attack', '9/11 related illness', 'Explosion', 'Bomb'])]

plt.figure(figsize=(15,8))

ts=ter.cause_short.value_counts()

sns.barplot(ts.index,ts.values,palette='cividis')

plt.xticks(rotation=70)

plt.xlabel('Cause of deaths by terrorist attacks')

plt.ylabel('Deaths')

plt.title('Terrorist Attacks')

plt.show()
plt.figure(figsize=(15,8))

tss=ter.state.value_counts()#.plot(kind='bar',color='slateblue')

sns.barplot(tss.index,tss.values,palette='gnuplot')

plt.xticks(rotation=80)

plt.xlabel('States')

plt.ylabel('Deaths')

plt.title('Terrorist deaths by states')

plt.show()
plt.figure(figsize=(15,8))

ter.year.value_counts().sort_index().plot(color='m')

plt.xlabel('Years')

plt.ylabel('Deaths')

plt.title('Terrorist deaths by Year')

plt.show()
plt.figure(figsize=(15,8))

ter.dept_name.value_counts().plot(color='rebeccapurple')

plt.xticks(rotation=30)

plt.xlabel('Departments')

plt.ylabel('Deaths')

plt.title('Terrorist deaths by Departments')

plt.show()
plt.figure(figsize=(15, 8))

wys=df[(df['cause_short']=='9/11 related illness') & (df['canine']==False)].year.value_counts().sort_index()

sns.barplot(wys.index,wys.values)

plt.ylabel("Deaths")

plt.xlabel("Years")

plt.title('9/11 Related Death by Year')

plt.show()
plt.figure(figsize=(15,6))

wss=df[(df['cause_short']=='9/11 related illness') & (df['canine']==False)].state.value_counts()

sns.barplot(wss.index,wss.values,palette='Set2')

plt.ylabel("Deaths")

plt.xlabel('States')

plt.title('9/11 Related Death by state')

plt.show()
plt.figure(figsize=(15, 8))

df[(df['cause_short']=='Gunfire') & (df['canine']==False)].year.value_counts().sort_index().plot( color = 'darksalmon')

plt.ylabel("Deaths")

plt.xlabel("Years")

plt.title('Gunfire Death by Year')

plt.show()
plt.figure(figsize=(15, 6))

df.groupby((df.year//10)*10).cause.count().plot(kind='bar')

plt.xlabel('Decades')

plt.ylabel('Deaths')

plt.title('Deaths by decades')

plt.show()
plt.figure(figsize=(15, 6))

df.groupby([(df['year']//100)*100]).cause.count().plot(kind='bar')

plt.show()
plt.figure(figsize=(15, 6))

df[(df['year']>=1900) & (df['year']<1901)].state.value_counts().plot(kind='bar')

plt.show()
plt.figure(figsize=(15, 6))

df[(df['year']>=1900) & (df['year']<1901)].cause_short.value_counts().plot(kind='bar')

plt.show()
df['ccode'] = df.cause_short.astype('category').cat.codes

df['scode'] = df.state.astype('category').cat.codes

df.head()
mdf=df[['year','scode','ccode']]

mdf.head()
year=mdf.drop('ccode',axis='columns')

year.head()
cause=mdf.ccode

cause.head()
from sklearn import linear_model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(year, cause)
knn.predict([[2020,59]])
mypre = [[1990,20], [2005,10],[2014,5],[2018,11]]
knn.predict(mypre)
df.ccode.max()
corr = df[['year','scode','ccode']].corr()

plt.figure(figsize=(15, 6))

sns.heatmap(corr)

plt.show()