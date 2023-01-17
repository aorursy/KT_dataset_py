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
import pandas as pd
df=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head()
df.duplicated().sum()
df.isna().sum()
df['Survived'].value_counts()
tempValues=df['Survived'].value_counts().values
labels=['Died','Survived']

import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(5,5),dpi=100)
explode=(0.1,0)
ax.pie(tempValues,labels=labels,autopct='%1.1f%%',explode=explode)
plt.title('Percentage population Died and Percentage population survived in the ship')
plt.show()
#Population of different countries in the ship

df['Country'].value_counts()
#Percentage population of top 5 populous country in the ship:

tempValues=df['Country'].value_counts().values[:5]
tempKeys=df['Country'].value_counts().keys()[:5]

fig,ax=plt.subplots(figsize=(5,5),dpi=100)
ax.pie(tempValues,labels=tempKeys,autopct='%1.1f%%')
plt.title('Representation of different countries in the ship')
plt.show()
#Mean age of survived and deceased population: 

df['Age'].groupby(df['Survived']).mean()
#Percentage of male-female population in the ship:

tempKeys=df['Sex'].value_counts().keys()
tempValues=df['Sex'].value_counts().values

fig,ax=plt.subplots(figsize=(5,5),dpi=100)
ax.pie(tempValues,labels=tempKeys,autopct='%1.1f%%')
plt.title('Percentage distribution of Male-female in the ship')
plt.show()
#Number of people survived for the different countries

df[df['Survived']==1]['Country'].value_counts()

tempkeys=df[df['Survived']==1]['Country'].value_counts().keys()
tempValues=df[df['Survived']==1]['Country'].value_counts().values
plt.bar(tempkeys,tempValues)
plt.tick_params(axis='x',rotation=70)
plt.title('Number of survived people across different countries')
plt.show()
#percentage wise distribution of survived population across different countries:

fig,ax=plt.subplots(figsize=(5,5),dpi=100)
ax.pie(tempValues,labels=tempkeys,autopct='%1.1f%%')
plt.title('% of different countries in the survived population')
plt.show()
#Male-female ratio in the survived population:

tempKeys=['Male','Female']
tempValues=df[df['Survived']==1]['Sex'].value_counts().values

fig,ax=plt.subplots(figsize=(5,5),dpi=100)
ax.pie(tempValues,labels=tempKeys,autopct='%1.1f%%')
plt.title('Male-female percentage in the survived population')
plt.show()
# Different 'Category' in the ship:

tempKeys=df['Category'].value_counts().keys()
tempValues=df['Category'].value_counts().values

plt.bar(tempKeys,tempValues)
plt.title('Total number of people in the different categories present in the ship')
plt.show()
#Percentage wise distribution of survived population in the different 'category':

tempKeys=df[df['Survived']==1]['Category'].value_counts().keys()
tempValues=df[df['Survived']==1]['Category'].value_counts().values

fig,ax=plt.subplots(figsize=(5,5),dpi=100)
ax.pie(tempValues, labels=tempKeys,autopct='%1.1f%%')
plt.title('Percentage wise distribution of the survived population according to \'category\'')
plt.show()
import seaborn as sns
sns.distplot(df['Age'])
plt.title('Distribution of Age')
plt.title('Distribution of the entire population onboard')
plt.show()
tempValues=df[df['Survived']==1]['Age']
tempValues2=df[df['Survived']==0]['Age']
ax=sns.distplot(tempValues,label='For Survived')
ax=sns.distplot(tempValues2,label='For Deceased')
ax.legend()
plt.title('Distribution of Age for the Deceased population and the survived population')
plt.show()
#Categorical plot for the 'category' criteria

sns.catplot(x='Category',y='Age',data=df,kind='violin',hue='Sex',split=True)
plt.show()
#Categorical plot for the 'Gender' criteria

sns.catplot(x='Sex',y='Age',data=df)
plt.show()
#Age distribution of 'Male' and 'Female' population in the ship

tempValuesMale=df[df['Sex']=='M']['Age']
tempValuesFemale=df[df['Sex']=='F']['Age']

ax=sns.distplot(tempValuesMale,label='Male')
ax=sns.distplot(tempValuesFemale,label='Female')
ax.legend()
plt.show()

tempMale=[]
tempFemale=[]
for i in range(len(df['Age'])):
    if df['Survived'][i]==1 and df['Sex'][i]=='M':
        tempMale.append(df['Age'][i])
    if df['Survived'][i]==1 and df['Sex'][i]=='F':
        tempFemale.append(df['Age'][i])
        

import numpy as np        
#Mean age of the survived Male population:
temp=np.mean(tempMale)
print(f'Mean age of the survived male population is {temp}')

#Mean age of the survived Female population:
temp=np.mean(tempFemale)
print(f'Mean age of the survived female population is {temp}')
#Distribution of the age of survived population with respect to the gender

ax=sns.distplot(tempMale,label='Male')
ax=sns.distplot(tempFemale,label='Female')
ax.legend()
plt.title('Distribution of the age of survived population with respect to the gender')
plt.xlabel('Age')
plt.show()
#Categorical plot for the age of passengers 
#boarding from different countries

sns.catplot(x='Country',y='Age',data=df)
plt.tick_params(axis='x',rotation=70)
plt.show()
sns.catplot(x='Sex',y='Survived',data=df,kind='point',hue='Category')
plt.show()
