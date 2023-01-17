# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
df15=pd.read_csv('../input/world-happiness-report/2015.csv')

df16=pd.read_csv('../input/world-happiness-report/2016.csv')

df17=pd.read_csv('../input/world-happiness-report/2017.csv')

df18=pd.read_csv('../input/world-happiness-report/2018.csv')

df19=pd.read_csv('../input/world-happiness-report/2019.csv')

df20=pd.read_csv('../input/world-happiness-report/2020.csv')

fx,ax=plt.subplots(3,2,figsize=(20,25))

sns.barplot(x=df15['Country'].head(10),y='Happiness Score',data=df15,ax=ax[0,0])

ax[0,0].set_title('Top 10 countries based on Happiness Score 2015',fontweight="bold")

ax[0,0].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df16['Country'].head(10),y='Happiness Score',data=df16,palette='husl',ax=ax[0,1])

ax[0,1].set_title('Top 10 countries based on Happiness Score 2016',fontweight="bold")

ax[0,1].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df17['Country'].head(10),y='Happiness.Score',data=df17,palette='spring',ax=ax[1,0])

ax[1,0].set_title('Top 10 countries based on Happiness Score 2017',fontweight="bold")

ax[1,0].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df18['Country or region'].head(10),y='Score',data=df18,palette='autumn',ax=ax[1,1])

ax[1,1].set_title('Top 10 countries based on Happiness Score 2018',fontweight="bold")

ax[1,1].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df19['Country or region'].head(10),y='Score',data=df19,ax=ax[2,0])

ax[2,0].set_title('Top 10 countries based on Happiness Score 2019',fontweight="bold")

ax[2,0].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df20['Country name'].head(10),y='Ladder score',data=df20,palette='Paired',ax=ax[2,1])

ax[2,1].set_title('Top 10 countries based on Ladder Score 2020',fontweight="bold")

ax[2,1].tick_params(axis='x', labelrotation=45)



plt.show()
fx,ax=plt.subplots(3,2,figsize=(20,25))

sns.barplot(x=df15['Country'].tail(10),y='Happiness Score',data=df15,ax=ax[0,0])

ax[0,0].set_title('Bottom 10 countries based on Happiness Score 2015',fontweight="bold")

ax[0,0].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df16['Country'].tail(10),y='Happiness Score',data=df16,palette='husl',ax=ax[0,1])

ax[0,1].set_title('Bottom 10 countries based on Happiness Score 2016',fontweight="bold")

ax[0,1].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df17['Country'].tail(10),y='Happiness.Score',data=df17,palette='spring',ax=ax[1,0])

ax[1,0].set_title('Bottom 10 countries based on Happiness Score 2017',fontweight="bold")

ax[1,0].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df18['Country or region'].tail(10),y='Score',data=df18,palette='autumn',ax=ax[1,1])

ax[1,1].set_title('Bottom 10 countries based on Happiness Score 2018',fontweight="bold")

ax[1,1].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df19['Country or region'].tail(10),y='Score',data=df19,ax=ax[2,0])

ax[2,0].set_title('Bottom 10 countries based on Happiness Score 2019',fontweight="bold")

ax[2,0].tick_params(axis='x', labelrotation=45)

sns.barplot(x=df20['Country name'].tail(10),y='Ladder score',data=df20,palette='Paired',ax=ax[2,1])

ax[2,1].set_title('Bottom 10 countries based on Ladder Score 2020',fontweight="bold")

ax[2,1].tick_params(axis='x', labelrotation=45)



plt.show()


df20.head(20).groupby('Regional indicator').agg({'Country name':'count'}).sort_values(by='Country name',ascending=False)
plt.figure(figsize=(10,5))

df20.head(20).groupby('Regional indicator').agg({'Country name':'count'}).sort_values(by='Country name',ascending=False).plot(kind='bar',color='g')

plt.show()
 

df20.tail(20).groupby('Regional indicator').agg({'Country name':'count'}).sort_values(by='Country name',ascending=False)
plt.figure(figsize=(10,5))

df20.tail(20).groupby('Regional indicator').agg({'Country name':'count'}).sort_values(by='Country name',ascending=False).plot(kind='bar',color='r')

plt.show()
cols=['Explained by: Log GDP per capita', 'Explained by: Social support',

       'Explained by: Healthy life expectancy',

       'Explained by: Freedom to make life choices',

       'Explained by: Generosity', 'Explained by: Perceptions of corruption',

       'Dystopia + residual']
for a in cols:

    plt.figure(figsize=(10,5))

    sns.regplot(x=a,y='Ladder score',data=df20,color='r')

    plt.show()
corr=df20[['Explained by: Log GDP per capita', 'Explained by: Social support',

       'Explained by: Healthy life expectancy',

       'Explained by: Freedom to make life choices',

       'Explained by: Generosity', 'Explained by: Perceptions of corruption',

       'Dystopia + residual','Ladder score']].corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr,annot=True)

plt.show()
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
y=df20['Ladder score']

X=df20[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption', 'Ladder score in Dystopia']]

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=1)
x_const=sm.add_constant(X)

model=sm.OLS(y,x_const).fit()

model.summary()
lr=LinearRegression()

lr.fit(X_train,Y_train)

print(f'R^2 score for train: {lr.score(X_train, Y_train)}')

print(f'R^2 score for test: {lr.score(X_test, Y_test)}')

y_pred_test=lr.predict(X_test)

y_pred_train=lr.predict(X_train)

print(f'mean squared error train: {mean_squared_error(Y_train,y_pred_train)}')

print(f'mean squared error test: {mean_squared_error(Y_test,y_pred_test)}')     
y=df20['Ladder score']

X=df20[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',

       'Freedom to make life choices','Perceptions of corruption', 'Ladder score in Dystopia']]

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=1)
x_const=sm.add_constant(X)

model=sm.OLS(y,x_const).fit()

model.summary()
lr=LinearRegression()

lr.fit(X_train,Y_train)

print(f'R^2 score for train: {lr.score(X_train, Y_train)}')

print(f'R^2 score for test: {lr.score(X_test, Y_test)}')

y_pred_test=lr.predict(X_test)

y_pred_train=lr.predict(X_train)

print(f'mean squared error train: {mean_squared_error(Y_train,y_pred_train)}')

print(f'mean squared error test: {mean_squared_error(Y_test,y_pred_test)}')     
df15['year']=2015

d15=df15[['Country', 'Happiness Score',

        'Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',

       'Generosity','year']]

df16['year']=2016

d16=df16[['Country', 'Happiness Score',

       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',

       'Freedom', 'Trust (Government Corruption)', 'Generosity','year']]

df17['year']=2017

d17=df17[['Country', 'Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',

       'Health..Life.Expectancy.', 'Freedom',

       'Trust..Government.Corruption.','Generosity','year']]

df18['year']=2018

d18=df18[['Country or region', 'Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices','Perceptions of corruption', 'Generosity','year']]

df19['year']=2019

d19=df19[[ 'Country or region', 'Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices','Perceptions of corruption', 'Generosity','year']]

df20['year']=2020

d20=df20[['Country name', 'Ladder score','Explained by: Log GDP per capita', 'Explained by: Social support',

       'Explained by: Healthy life expectancy',

       'Explained by: Freedom to make life choices','Explained by: Perceptions of corruption',

       'Explained by: Generosity','year' ]]

d20.head()
## Changing the names of the variables to make it uniform across datasets

a=d15.columns

b=d17.columns

for c,d in zip(a,b):

    d17=d17.rename(columns={d:c})

    

e=d18.columns

for c,d in zip(a,e):

    d18=d18.rename(columns={d:c})

    

f=d19.columns

for c,d in zip(a,f):

    d19=d19.rename(columns={d:c})

    

g=d20.columns

for c,d in zip(a,g):

    d20=d20.rename(columns={d:c})
df=pd.concat([d15,d16,d17,d18,d19,d20],axis=0)

df.head()
df.tail()
countries=['India','United States','United Kingdom','Russia','China','Canada','Germany','France','Switzerland', 'Iceland', 'Denmark', 'Norway', 'Finland',

       'Netherlands','Japan', 'South Korea','Italy','Singapore','Greece','Iran','Spain','Mexico','Egypt','Ukraine', 'Iraq', 'South Africa']

df1=df[df['Country'].isin(countries)]

 

import plotly.express as px

fig=px.line(df1,x='year',y='Happiness Score',color='Country',template="plotly_dark")

fig.show()
countries=['India','United States','United Kingdom','Russia','China','Canada','Germany','France','Switzerland','United Arab Emirates','Japan','Italy'] 

df2=df[df['Country'].isin(countries)]

fig=px.line(df2,x='year',y='Happiness Score',color='Country',template="plotly_dark",title='Year vs. Happiness Score')

fig.show()
fig=px.line(df2,x='year',y='Economy (GDP per Capita)',color='Country',template="plotly_dark",title='Year vs. Economy (GDP per Capita) ')

fig.show()
fig=px.line(df2,x='year',y='Health (Life Expectancy)',color='Country',template="plotly_dark",title='Year vs. Health (Life Expectancy)')

fig.show()
fig=px.line(df2,x='year',y='Family',color='Country',template="plotly_dark",title='Year vs. Family')

fig.show()
fig=px.line(df2,x='year',y='Trust (Government Corruption)',color='Country',template="plotly_dark",title='Year vs. Trust (Government Corruption)')

fig.show()
fig=px.line(df2,x='year',y='Freedom',color='Country',template="plotly_dark",title='Year vs. Freedom')

fig.show()
fig=px.line(df2,x='year',y='Generosity',color='Country',template="plotly_dark",title='Year vs. Generosity')

fig.show()