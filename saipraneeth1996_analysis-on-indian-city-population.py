#Importing essential libraries for data cleaning and Visualization

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns



import os

print(os.listdir("../input"))
#reading the data present in csv file to a dataframe 

df=pd.read_csv('../input/cities_r2.csv')

df.head().T
#Renaming columns of data set for easy manipulation

df.columns=['city','scode','state','dcode','poptotal','popmale','popfemale','0-6poptotal','0-6popmale','0-6popfemale',

            'litertotal','litermale','literfemale','sexratio','0-6sexratio','elrtotal','elrmale','elrfemale',

            'location','totalgrad','malegrad','femalegrad']
#Checking for null values in the dataframe

df.isnull().sum()
#getting metadata of the dataset

df.info()
df.describe().T
city_pop = df['poptotal'].sum()

total_indian_pop = 1247200000 #according to 2011 census

percentage = (city_pop/total_indian_pop)*100

print('{0:.1f}% of population living in cities'.format(percentage))
df['gradratio'] = df['totalgrad']/df['poptotal']

grad_avg = df['gradratio'].mean()

print('Graduate Ratio in Indian Cities: {0:.2f}'.format(grad_avg))
df['latitude'] = df['location'].apply(lambda x: x.split(',')[0])

df['longitude'] = df['location'].apply(lambda x: x.split(',')[1])
df = df.drop('location',axis = 1)
df.head().T
#Total no. of states

df.state.nunique()
#Total no. of cities

df.city.count()
plt.figure(figsize=(12,8))

city_count = df.groupby('state')['city'].count().sort_values(ascending=True)

city_count.plot(kind="barh", fontsize = 10,width=1,edgecolor='black')

plt.xlabel('No of cities', fontsize =15)

plt.ylabel('States', fontsize =15)

plt.title('Count of Cities taken from each State', fontsize = 20)

plt.show()
#STATES WITH THEIR CORRESPONDING CODES



sc = df[['state','scode']].groupby(['state','scode']).count()

print('State codes')

sc
#NO. OF DISTRICTS PRESENT IN EACH STATE
dc=df[['state','dcode']].groupby(['state']).count()

dc=dc.sort_values(by='dcode',ascending=False)

print('No. of districts present in each state')

dc
plt.figure(figsize=[14,10])

sns.barplot(y=dc.index, x=dc.dcode)

plt.title('Number of Districts in each States Taken',fontsize=25)

plt.ylabel('State', fontsize = 20)

plt.xlabel('No. of Districts', fontsize = 20)

plt.show()
#TOP 10 POPULATED STATES

population_state = df[['state','poptotal']].groupby('state').sum().sort_values(by='poptotal',ascending=False)

population_state.head(10)
#LEAST 10 POPULATED STATES

population_state.tail(10)
plt.figure(figsize=[14,10])

sns.barplot(y=population_state.index, x=population_state.poptotal)

plt.title('States according to Total Population',fontsize=25)

plt.xlabel('Population', fontsize = 20)

plt.ylabel('State', fontsize = 20)

plt.show()
#TOP 10 POPULATED CITIES

df[['city','poptotal']].sort_values(by='poptotal',ascending=False).head(10)
#LEAST 10 POPULATED CITIES

df[['city','poptotal']].sort_values(by='poptotal',ascending=False).tail(10)
df.popmale.sum()
df.popfemale.sum()
df['0-6popmale'].sum()
df['0-6popfemale'].sum()
df[df.popmale==df.popmale.max()]
df[df.popfemale==df.popfemale.max()]
#TOP 10 CHILD POPULATED CITIES



df[['city','0-6poptotal']].sort_values(by='0-6poptotal',ascending=False).head(10)
#LEAST 10 CHILD POPULATED CITIES



df[['city','0-6poptotal']].sort_values(by='0-6poptotal',ascending=False).tail(10)
df.columns
#TOP 10 CITIES WITH HIGH LITERACY



df[['city','elrtotal']].sort_values(by='elrtotal',ascending=False).head(10)
#TOP 10 CITIES WITH LEAST LITERACY





df[['city','elrtotal']].sort_values(by='elrtotal',ascending=False).tail(10)
lit_states  = df.groupby('state').agg({'litertotal': np.sum})

popstates  = df.groupby('state').agg({'poptotal': np.sum})



literate_rate = lit_states.litertotal * 100 / popstates.poptotal

literate_rate = literate_rate.sort_values(ascending=False)
literate_rate
plt.figure(figsize=[14,10])

sns.barplot(x=literate_rate, y=literate_rate.index)

plt.title('States according to literacy rate',fontsize=25)

plt.xlabel('Literacy Rate', fontsize = 20)

plt.ylabel('State', fontsize = 20)

plt.show()
#TOP 10 CITIES WITH HIGH SEX RATIO



df[['city','sexratio']].sort_values(by='sexratio',ascending=False).head(10)
#TOP 10 CITIES WITH LEAST SEX RATIO



df[['city','sexratio']].sort_values(by='sexratio',ascending=False).tail(10)
sex_ratio_states=df.groupby('state')['sexratio'].mean().sort_values(ascending=False)



plt.figure(figsize=[14,10])

sex_ratio_states.plot(kind='barh',edgecolor='black',width=0.9)

plt.title('States according to sex ratio',fontsize=25)

plt.xlabel('Sex Ratio', fontsize = 20)

plt.ylabel('State', fontsize = 20)

plt.show()
grad_ratio_states=df.groupby(['state'])['gradratio'].mean().sort_values(ascending=False)



plt.figure(figsize=(25, 10))

grad_ratio_states.plot(kind='bar',fontsize=15,width=0.9,edgecolor='black')

plt.title('Graduates Ratio of the States considering all the Top Cities in it',fontsize=30)

plt.ylabel('Graduate Ratio',fontsize = 20)
df['gradratio']= (df.gradratio-df.gradratio.min())/(df.gradratio.max()-df.gradratio.min())
df['sexratio']= (df.sexratio-df.sexratio.min())/(df.sexratio.max()-df.sexratio.min())
df['elrtotal']= (df.elrtotal-df.elrtotal.min())/(df.elrtotal.max()-df.elrtotal.min())
df['parameter'] = df.gradratio + df.sexratio + df.elrtotal
best_state=df[['state','parameter']].groupby('state').mean().sort_values(by='parameter', ascending=False)

best_state.head(10)
plt.figure(figsize=[14,10])

sns.barplot(x=best_state.parameter, y=best_state.index)

plt.title('Best States Overall based on created Parameter',fontsize=25)

plt.xlabel('Parameter', fontsize = 20)

plt.ylabel('State', fontsize = 20)

plt.show()