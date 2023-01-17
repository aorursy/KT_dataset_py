#Import Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/fifa19/data.csv')



#Checking first 10 features

data.head(10)
#Number of Null Values.

data.isnull().sum().max()
data.columns
data.describe()
#Look for correlation between Age and Potential or Overall and Potential.

data.corr()
f,ax=plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),annot=True,linewidth=1,fmt='.1f',ax=ax)

plt.show()
#Lets look at the graph for see better

data.plot(kind='scatter',x='Age',y='Potential',color='blue',alpha=0.5,figsize=(10,10))

data.plot(kind='scatter',x='Overall',y='Potential',color='red',alpha=0.5,figsize=(10,10))

plt.show()
data.Age.plot(kind='hist',bins=50)

plt.xlabel('Age')

plt.ylabel('Number')

plt.show()
#Lets Check The Young Talents

def age(x):

    return data[data['Age']==x][['Name','Club','Position','Overall','Potential','Nationality','Value']]

age(20)
def country(x):

    return data[data['Nationality']==x][['Name','Club','Position','Overall','Potential','Age','Value']]

tr=country('Turkey')

tr.head(20)
tr.tail(25)
tr.shape
tr.describe()
tr.index
# Look for TR Clubs but I'm transforming to set because there're too many of the same team

players_club = set(tr['Club'])

players_club
superleague_teams=['Kasimpaşa SK','BB Erzurumspor','Beşiktaş JK','Göztepe SK','Trabzonspor','Galatasaray SK','Medipol Başakşehir FK','Alanyaspor','Yeni Malatyaspor','Çaykur Rizespor','Sivasspor','MKE Ankaragücü','Atiker Konyaspor','Bursaspor','Antalyaspor']

#type(superleague_teams)
#We take all turkish players in the Super League

tr_league=tr[tr['Club'].isin(superleague_teams)]

tr_league
# We can see the average age of all tr clubs.

tr_league.groupby('Club')['Age'].mean()
tr_league.plot(kind='scatter',x='Age',y='Overall',alpha=0.5,color='r',figsize=(13,13))

plt.xlabel('Age')

plt.ylabel('Overall')

plt.show()
tr_league.plot(kind='scatter',x='Overall',y='Potential',alpha=0.5,color='r',figsize=(13,13))

plt.xlabel('Overall')

plt.ylabel('Potential')

plt.show()