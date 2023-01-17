import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#df1993=pd.read_csv('US_births_1994-2003_CDC_NCHS.csv')

#df3=df1993[df1993.year<2000]

#

#df3.shape
df1=pd.read_csv("../input/us-births19942014/US_births_1994-2003_CDC_NCHS.csv").query('year<2000')

df1.shape
df1
df2=pd.read_csv('../input/us-births19942014/US_births_2000-2014_SSA.csv')

df2.shape
df=pd.concat([df1,df2],ignore_index=True)

df.shape
df.head()
df.isnull().sum()#checking for is there any null values
df['day_of_week'].replace([1,2,3,4,5,6,7],['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True)

df.day_of_week.value_counts()
df['month'].replace([1,2,3,4,5,6,7,8,9,10,11,12],['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December']

,inplace=True)

df.month.value_counts()
df.nunique()
sns.set_style('whitegrid')

sns.set_context('talk')

plt.figure(figsize=(8,4))

sns.distplot(df['births'])

plt.show()
plt.figure(figsize=(10,4))

df.groupby('day_of_week').births.mean().plot(kind='bar',color=['black', 'red', 'green', 'blue', 'cyan','m','yellow'])

plt.show()
plt.figure(figsize=(10,15))

df.groupby('day_of_week').births.mean().plot(kind='pie',label='',autopct='%1.1f%%',explode=[0,0,0,0.2,0,0.1,0],shadow=True)#,colors=['black', 'red', 'green', 'blue', 'cyan','m','yellow'])

plt.show()
plt.figure(figsize=(10,8))

sns.stripplot(x="births", y="day_of_week", data=df,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.show()
plt.figure(figsize=(10,6))

sns.barplot(x='day_of_week',y='births',data=df,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.show()
plt.figure(figsize=(10,8))

sns.boxplot(x='day_of_week',y='births',data=df,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.show()
plt.figure(figsize=(10,8))

sns.violinplot(x="births", y="day_of_week", data=df,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x="births", y="month",order=['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December'],data=df,palette='winter')

plt.show()
plt.figure(figsize=(10,10))

df.groupby('month').births.mean().plot(kind='pie',label='',autopct='%1.1f%%',shadow=True,explode=[0,0,0,0,0.2,0,0,0,0,0,0,0.3])

plt.show()
plt.figure(figsize=(20,10))

sns.lineplot(x='month',y='births',hue='day_of_week',hue_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],data=df,estimator='mean',sort=False)

plt.show()
plt.figure(figsize=(15,10))

sns.stripplot(x="births", y="month",hue='day_of_week',hue_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],order=['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December'] ,data=df)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x="births", y="month",order=['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December'],hue='day_of_week',hue_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],data=df)

plt.show()
plt.figure(figsize=(12,6))

df.groupby('year').births.mean().plot(kind='bar',color='m')

plt.show()
plt.figure(figsize=(12,6))

df.groupby('year').births.mean().plot(kind='line',color='g',linewidth=1.5,marker='*',markersize=10)

plt.show()
plt.figure(figsize=(15,10))

sns.lineplot(x='year',y='births',data=df,hue='month')#hue_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],data=g)

plt.show()
plt.figure(figsize=(20,10))

sns.scatterplot(x='year', y='births', data=df,hue='month')

plt.show()
df10=df[(df['year']==1994)|(df['year']==2004)|(df['year']==2014)]

df10.nunique()#Verifiing the DataFrame
#df10=pd.concat([df[df.year==1994],df[df.year==2004],df[df.year==2014]],ignore_index=True)

#df10.shape
#sdf=df10.groupby(['year','month','day_of_week'],as_index=False).births.mean()#verifying the dataframe

#sdf.nunique()
plt.figure(figsize=(15,10))

plt.title("Changes in births rate over decades of time")

df10.groupby('year').births.mean().plot(kind='pie',label='',autopct='%1.1f%%',explode=[0,0.1,0],shadow=True)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x='year',y='births',data=df10,palette='mako')

plt.show()
plt.figure(figsize=(15,10))

sns.stripplot(x='year',y='births',data=df10)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x='year',y='births',hue='day_of_week',hue_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],data=df10)

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(x='year',y='births',data=df10,hue='month',palette='tab20')

plt.show()
friday=df[(df.date_of_month==13) & (df.day_of_week=='Friday')]

friday.births.sum()
plt.figure(figsize=(20,10))

sns.barplot(x='year', y='births',data=friday,palette='ocean')

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x="births", y="month",order=['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December'],data=friday,palette='plasma_r')

plt.show()
christmas=df[((df.date_of_month==25) & (df.month=='December'))]

christmas.births.sum()
plt.figure(figsize=(20,10))

sns.barplot(x='year', y='births',data=christmas,palette='Spectral')

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(x='day_of_week', y='births',data=christmas,palette='cubehelix')

plt.show()
newyear=df[(df.date_of_month==1)  & (df.month=='January')]

newyear.births.sum()
plt.figure(figsize=(20,10))

sns.barplot(x='year', y='births',data=newyear,palette='winter')

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(x='day_of_week', y='births',data=newyear,palette='rocket')

plt.show()
plt.figure(figsize=(20,10))

plt.pie(x=[friday.births.mean(),christmas.births.mean(),newyear.births.mean()],labels=['Friday the 13th','Christamas','New Year'],autopct='%1.2f%%',explode=[0.1,0,0],shadow=True,colors=['orange','pink','g'])

plt.show()