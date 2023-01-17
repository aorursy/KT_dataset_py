import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(30,15)})
ply_str_app=pd.read_csv('../input/googleplaystore.csv')
ply_str_userR=pd.read_csv('../input/googleplaystore_user_reviews.csv')
ply_str_app.drop_duplicates('App',keep='first',inplace=True)
ply_str_app.drop(index=10472,inplace=True)

ply_str_app.head()
ply_str_userR.head()
ply_str_app['Category'].unique()
sns.catplot(
y='Category',
    data=ply_str_app,
    kind='count',height=30
)
ply_str_app['Type'].value_counts()
ply_str_app[ply_str_app['Type'].isnull()]
ply_str_app['Type'][9148]='Free'
ply_str_app.isnull().sum()
ply_str_app.dtypes
ply_str_app['Reviews']=ply_str_app['Reviews'].astype('int')
sns.jointplot(
x='Rating',
    y='Reviews',
    data=ply_str_app,height=20)
ply_str_app['Installs'].value_counts()
ply_str_app['Installs']=ply_str_app['Installs'].str.replace('+','')
ply_str_app['Installs']=ply_str_app['Installs'].str.replace(',','')
ply_str_app['Installs'].value_counts()
ply_str_app['Installs']=ply_str_app['Installs'].astype(int)
sns.jointplot(
y='Installs',
    x='Rating',
    data=ply_str_app,
    height=25
)
sns.set(rc={'figure.figsize':(30,15)})
sns.boxplot(
y='Rating',
    x='Installs',
    data=ply_str_app,
)
ply_str_app['Price']=ply_str_app['Price'].str.replace('$','')
ply_str_app['Price']=ply_str_app['Price'].astype(float)
ply_str_app.Price.unique()
sns.distplot(ply_str_app['Price'])

df=ply_str_app[['Price','Installs','Rating','Reviews',]]
sns.heatmap(df.corr(),linewidths=1,annot=True)




