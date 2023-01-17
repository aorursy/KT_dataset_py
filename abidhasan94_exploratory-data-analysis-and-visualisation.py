import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
x = pd.read_csv('../input/googleplaystore.csv');x
x.groupby('Type')['Category'].count()
x.drop(['Current Ver','Price'],axis=1,inplace=True)
x.info()
x = x[x['Content Rating'].notna()]
len(x['Category'].unique())
x['Content Rating'].isna().any()
print(x.groupby(['Category'])['Rating'].mean())
x['Rating']=x.groupby(['Category','Content Rating'])['Rating'].transform(lambda z: round(z.fillna(z.mean()),1) )
x['Rating']=x.groupby(['Category'])['Rating'].transform(lambda z: round(z.fillna(z.mean()),1) )
#Check if there are still any NaN values
x[x['Rating'].isna()]
x.info()
x.groupby(['Category','Content Rating'])['Rating'].mean()
x['Content Rating'].describe()
x_cnt = pd.DataFrame(x.groupby('Category')['Type'].count()).sort_values(by='Type',ascending=False)
x_cnt.columns = ['Count']
x_cnt
x['Installs'].describe()
x['Installs'] = x['Installs'].str.replace('[+,]','')
x['Installs']=pd.to_numeric(x['Installs'],errors='coerce')
x['Reviews']=pd.to_numeric(x['Reviews'],errors='coerce')
x['Rating'].unique()
sns.catplot(x='Content Rating',y='Rating',hue='Type',aspect=2,height=7,data=x)
x.info()
x['Installs'].describe()
z = x['Installs']
x['Rating']=x.groupby(['Category'])['Rating'].transform(lambda z: round(z.fillna(z.mean()),1) )
x.info()
x['Installs'].unique()
x
x_cnt = pd.DataFrame(x.groupby('Category')['Type'].count()).sort_values(by='Type',ascending=False)
x_cnt.columns=['Count']
x_cnt

x
g = sns.catplot(y='Category', x= 'Rating',kind='box',aspect=1.5,height=10, data=x)

g = sns.catplot(y='Category', x= 'Rating',kind='boxen',aspect=1.5,height=10, data=x)

g = sns.catplot(y='Category', x= 'Rating',kind='violin',aspect=1.5,height=10, data=x)


x['Age'] = (pd.to_datetime('2018-09-20')-pd.to_datetime(x['Last Updated'])).dt.days
x['Age']
g =sns.catplot(y='Category',x='Age',aspect=1.5,height=15,data=x)

x.groupby(['Installs'])['Age'].mean()
x.groupby(['Installs'])['Age'].std().plot()
x.groupby(['Installs'])['Rating'].std().plot()

