import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import Series, DataFrame
df1=pd.read_csv('../input/tipping/tips.csv')
df1.head()
df1.tail()
df1.columns
df1.info()
df1.isnull().sum()
df1.describe()
df1['day'].unique()
df1.nunique()
for i in df1.columns:

    print(df1[i].unique())
a=pd.DataFrame(df1['day'].value_counts())

a.reset_index(inplace=True)

plt.bar(a['index'],a['day'])
plt.bar(df1['day'].value_counts().index,df1['day'].value_counts().values)
a.plot(kind='bar',x='index',y='day',colormap='icefire')

plt.xticks(rotation=0)

plt.show()
sns.barplot(a['index'],a['day'])

plt.show()
plt.pie(a['day'],labels=a['index'],autopct='%1.2f',explode=[0.2,0,0,0])

plt.show()
a.plot(kind='pie',y='day',labels=a['index'],autopct='%1.2f')

plt.show()
sns.distplot(df1['total_bill'])
a=df1['total_bill']

mean=a.mean()

median=np.median(a)

mode=a.mode()
sns.distplot(a,hist=False)

plt.axvline(mean,color='r',label='mean')

plt.axvline(median,color='b',label='median')

plt.axvline(mode[0],color='g',label='mode')

plt.legend()

plt.show()
a.plot(kind='kde')

plt.axvline(mean,color='r',label='mean')

plt.axvline(median,color='b',label='median')

plt.axvline(mode[0],color='g',label='mode')

plt.legend
a.plot(kind='density')

plt.axvline(mean,color='r',label='mean')

plt.axvline(median,color='b',label='median')

plt.axvline(mode[0],color='g',label='mode')

plt.legend
plt.boxplot(a)

plt.text(0.85,13,s='Q1',size=13)

plt.text(0.85,17,s='Q2',size=13)

plt.text(0.85,23,s='Q3',size=13)

plt.text(1.1,16,s='IQR',rotation=90,size=20)

plt.show()
sns.boxplot(a,color='Purple')

plt.show()
plt.hist(df1['total_bill'],color='orange',bins=[10,15,25,30,50],edgecolor='black',rwidth=0.5)

plt.show()
sns.violinplot(a,color='Yellow')

plt.show()
a.kurt()
a.skew()
x1=[1,2,3,4]

y1=[5,10,3,20]

x2=[10,15,12,11]

plt.plot(x1,y1,linestyle='-.',marker='^',markersize=9,color='green',label='line 1')

plt.plot(x1,x2,linestyle='--',marker='o',markersize=9,color='red',label='line 2')

plt.title('Simple Line Graph')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.legend()

plt.grid()

plt.show()
plt.scatter(df1['total_bill'],df1['tip'])

plt.show()
sns.scatterplot(x='total_bill',y='tip',data=df1,hue='day')

plt.show()
sns.scatterplot(x='total_bill',y='tip',data=df1,hue='sex')

plt.show()
sns.lmplot(x='total_bill',y='tip',data=df1,hue='sex',fit_reg=False,markers=['^','s'],palette='ocean',row='sex',col='smoker')

# col and row only for LM plot

plt.show()
sns.stripplot(x='day',y='total_bill',data=df1,jitter=False)

plt.grid()

plt.axhline(20,color='black')

plt.show()
sns.swarmplot(x='day',y='total_bill',data=df1,hue='sex')

plt.grid()

plt.axhline(20,color='black')

plt.show()
sns.heatmap(df1.corr(),annot=True)

plt.figure(figsize=(5,5))

plt.show()
sns.countplot(x='day',data=df1,hue='sex')

plt.show()
sns.countplot(x='sex',data=df1,hue='day')

plt.show()
sns.countplot(x='size',data=df1,hue='sex')

plt.show()
sns.pairplot(data=df1,hue='sex')

plt.show()
sns.boxplot(x='day',y='total_bill',data=df1)

plt.show()
sns.violinplot(y='day',x='total_bill',data=df1)

plt.show()
sns.boxplot(x='day',y='total_bill',data=df1,whis=False)

plt.show()
sns.violinplot(x='day',y='total_bill',data=df1,hue='smoker',split=True)

plt.show()
a=df1.groupby('sex').mean()['total_bill']
a.plot(kind='bar')

plt.show()
df1.groupby('day').mean()['total_bill'].plot(kind='bar')

plt.show()
x=pd.DataFrame(pd.pivot_table(df1,index=['sex','smoker'],aggfunc='count')['total_bill'])
x
x.loc['Female','Yes'].sum()/x.loc['Female'].sum()*100
x.loc['Female','No'].sum()/x.loc['Female'].sum()*100
df1['smoker'][df1['sex']=='Female'].value_counts(normalize=True)*100
df1[(df1['sex']=='Female') & (df1['smoker']=='Yes')]['sex'].value_counts()
(x.loc['Female','Yes'].sum()/(x.loc['Female','Yes'].sum()+x.loc['Male','Yes'].sum()))*100
(x.loc['Male','Yes'].sum()/(x.loc['Female','Yes'].sum()+x.loc['Male','Yes'].sum()))*100
print((df1.groupby(['day','smoker']).count()['total_bill']['Thur','Yes']/df1.groupby(['day','smoker']).count()['total_bill']['Thur'].sum())*100)
print((df1.groupby(['day','smoker']).count()['total_bill']['Fri','Yes']/df1.groupby(['day','smoker']).count()['total_bill']['Fri'].sum())*100)
print((df1.groupby(['day','smoker']).count()['total_bill']['Sat','Yes']/df1.groupby(['day','smoker']).count()['total_bill']['Sat'].sum())*100)

print((df1.groupby(['day','smoker']).count()['total_bill']['Sun','Yes']/df1.groupby(['day','smoker']).count()['total_bill']['Sun'].sum())*100)
print((df1.groupby(['day','smoker']).count()['total_bill']['Thur','No']/df1.groupby(['day','smoker']).count()['total_bill']['Thur'].sum())*100)

print((df1.groupby(['day','smoker']).count()['total_bill']['Fri','No']/df1.groupby(['day','smoker']).count()['total_bill']['Fri'].sum())*100)

print((df1.groupby(['day','smoker']).count()['total_bill']['Sat','No']/df1.groupby(['day','smoker']).count()['total_bill']['Sat'].sum())*100)

print((df1.groupby(['day','smoker']).count()['total_bill']['Sun','No']/df1.groupby(['day','smoker']).count()['total_bill']['Sun'].sum())*100)
(df1.groupby(['day','smoker']).count()['total_bill']/df1.groupby(['day']).count()['total_bill'])*100
df1['sex LE']=df1['sex'].replace({'Male':0,'Female':1})#replace returns numerical value as int type

df1['sex LE map']=df1['sex'].map({'Male':0,'Female':1})# map returns numeric values as category type

df1.head()
df1['sex LE map']=df1['sex LE map'].astype(np.int64)
df1.info()
from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
df1['LE Day']=lr.fit_transform(df1['day'])
df1.head()
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)

df1['sex'].value_counts().plot(kind='bar')

plt.subplot(2,2,2)

df1['sex'].value_counts().plot(kind='pie')

plt.show()
fig, axes=plt.subplots(1,2,figsize=(15,5))

df1['sex'].value_counts().plot(kind='bar',ax=axes[1])

df1['sex'].value_counts().plot(kind='pie',ax=axes[0])

plt.show()
df1['zscore']=(df1['total_bill']-df1['total_bill'].mean())/df1['total_bill'].std()
df2=df1[(df1['zscore']>3) | (df1['zscore']<-3) ]

df2
df3=df1[(df1['zscore']<3) & (df1['zscore']>-3)]

df3
df2['total_bill'].count()
plt.subplot(2,2,1)

df1['total_bill'].plot(kind='kde')

plt.subplot(2,2,2)

df3['total_bill'].plot(kind='kde')

plt.show()
print(df1['total_bill'].skew())

print(df3['total_bill'].skew())
print(df1['total_bill'].kurt())

print(df3['total_bill'].kurt())
q1=df1['total_bill'].quantile(0.25)

q2=df1['total_bill'].quantile(0.5)

q3=df1['total_bill'].quantile(0.75)
IQR=q3-q1

IQR
UL=q3+(IQR)*(3/2)

UL
LL=q1-(IQR)*(3/2)

LL
df_out=df1[(df1['total_bill']>UL) | (df1['total_bill']<LL)]

df_out['total_bill'].count()
df_clean=df1[(df1['total_bill']<=UL) & (df1['total_bill']>=LL)]

df_clean.head()
plt.figure(figsize=(15,10))

plt.subplot(3,3,1)

df1['total_bill'].plot(kind='kde')

plt.subplot(3,3,2)

df3['total_bill'].plot(kind='kde')

plt.subplot(3,3,3)

df_clean['total_bill'].plot(kind='kde')

plt.show()
print('skew for all data is',df1['total_bill'].skew())

print('skew for all clean data with zscore is',df3['total_bill'].skew())

print('skew for clean data is',df_clean['total_bill'].skew())
print('kurtosis for all data is',df1['total_bill'].kurt())

print('kurtosis for all clean data with zscore is',df3['total_bill'].kurt())

print('kurtosis for clean data is',df_clean['total_bill'].kurt())
print('Upper limit is',UL)

print('Lower limit is',LL)

print('IQR is',IQR)

df_mm=(df1['total_bill']-df1['total_bill'].max())/(df1['total_bill'].max()-df1['total_bill'].min())

df_mm
plt.figure(figsize=(15,10))

plt.subplot(3,3,1)

df1['total_bill'].plot(kind='kde')

plt.subplot(3,3,2)

df3['total_bill'].plot(kind='kde')

plt.subplot(3,3,3)

df_mm.plot(kind='kde')

plt.show()
print('min max',df_mm.skew())
print('kurtosis for all data is',df1['total_bill'].kurt())

print('kurtosis for all clean data with zscore is',df3['total_bill'].kurt())

print('kurtosis for all clean data with minmax is',df_mm.kurt())
df3['total_bill'].skew()


df1['sqrt']=(df1['total_bill'])**(1/2)

df1['log']=np.log(df1['total_bill'])
df1
plt.figure(figsize=(15,10))

plt.subplot(3,3,1)

df1['total_bill'].plot(kind='kde')

plt.xlabel('original')

plt.subplot(3,3,2)

df1['sqrt'].plot(kind='kde')

plt.xlabel('square root')

plt.subplot(3,3,3)

df1['log'].plot(kind='kde')

plt.xlabel('log')

plt.show()
print('skewness of original data',df1['total_bill'].skew())

print('skewness of root transform data',df1['sqrt'].skew())

print('skewness of log transform data',df1['log'].skew())
from sklearn.model_selection import train_test_split
x=df1.drop(['total_bill'],axis=1)

y=df1['total_bill']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=0)
pd.crosstab(df1['sex'],df1['day'])
pd.crosstab(df1['sex'],df1['day']).plot(kind='bar')

plt.show()

pd.crosstab(df1['sex'],df1['day']).plot(kind='bar',stacked=True)

plt.show()