import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data=pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')
data.head()
data.isnull().sum() #checking for total null values
data.dtypes #Checking features types
X, Y, S, A, Dis, Orientation, dir, NflId, JerseyNumber, Season, YardLine, Quarter, 

Down, Distance, HomeScoreBeforePlay, VisitorScoreBeforePlay, 

NflIdRusher, DefendersInTheBox, Yards, PlayerWeight, Week, Temperature, Humidity
f,ax=plt.subplots(1,19,figsize=(25,10))

data[['PlayerWeight','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Yards depending on PlayerWeight')

data[['Temperature','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[1])

ax[1].set_title('Yards depending on Temperature')

data[['X','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[2])

ax[2].set_title('Yards depending on X')

data[['Y','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[3])

ax[3].set_title('Yards depending on Y')

data[['S','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[4])

ax[4].set_title('Yards depending on S')

data[['A','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[5])

ax[5].set_title('Yards depending on A')

data[['JerseyNumber','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[6])

ax[6].set_title('Yards depending on JerseyNumber')

data[['Dis','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[7])

ax[7].set_title('Yards depending on Dis')

#data[['dir','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[8])

#ax[8].set_title('Yards depending on dir')

data[['NflId','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[9])

ax[9].set_title('Yards depending on NflId')

data[['YardLine','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[10])

ax[10].set_title('Yards depending on YardLine')

data[['Quarter','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[11])

ax[11].set_title('Yards depending on Quarter')

data[['Down','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[12])

ax[12].set_title('Yards depending on Down')

data[['Distance','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[13])

ax[13].set_title('Yards depending on Distance')

data[['HomeScoreBeforePlay','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[14])

ax[14].set_title('Yards depending on HomeScoreBeforePlay')

data[['VisitorScoreBeforePlay','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[15])

ax[15].set_title('Yards depending on VisitorScoreBeforePlay')

data[['Week','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[16])

ax[16].set_title('Yards depending on Week')

data[['Humidity','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[17])

ax[17].set_title('Yards depending on Humidity')

data[['DefendersInTheBox','Yards']].groupby(['Yards']).mean().plot.bar(ax=ax[18])

ax[18].set_title('Yards depending on DefendersInTheBox')



plt.show()
print('Longest Yards was of:',data['Yards'].max(),'yards')

print('Smalest Yards was of:',data['Yards'].min(),'yards')

print('Average Yards was of:',data['Yards'].mean(),'yards')
data['Age'] = data['PlayerBirthDate'].map(lambda x: 2018-int(x.split('/')[2]))
print('Oldest Player was of:',data['Age'].max(),'Years')

print('Youngest Player was of:',data['Age'].min(),'Years')

print('Average Age on the field:',data['Age'].mean(),'Years')
data['Experience'] = data['Age'].map(lambda x: 1 if x>25 else 0)
data['Experience'].head()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("JerseyNumber","Age", hue="Experience", data=data,split=True,ax=ax[0])

ax[0].set_title('JerseyNumber and Age vs Experience')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("A","Age", hue="Experience", data=data,split=True,ax=ax[1])

ax[1].set_title('Accelaration and Age vs Experience')

ax[1].set_yticks(range(0,110,10))

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("JerseyNumber","Yards", hue="Experience", data=data,split=True,ax=ax[0])

ax[0].set_title('JerseyNumber and Yards vs Experience')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("A","Yards", hue="Experience", data=data,split=True,ax=ax[1])

ax[1].set_title('Accelaration and Yards vs Experience')

ax[1].set_yticks(range(0,110,10))

plt.show()
data['PlayerCollegeName'].head()
print('There is '+str(len(set(data['PlayerCollegeName'].to_list())))+' different PlayerCollegeName')
data.groupby(['PlayerCollegeName','A'])['PlayerCollegeName'].count()
data.groupby(['PlayerCollegeName','S'])['PlayerCollegeName'].count()
data.groupby(['PlayerCollegeName','PlayerHeight'])['PlayerCollegeName'].count()
data.groupby(['PlayerCollegeName','PlayerWeight'])['PlayerCollegeName'].count()
data.groupby(['PlayerCollegeName','Yards'])['PlayerCollegeName'].count()
data['VisitorTeamAbbr'].head()
print('There is '+str(len(set(data['VisitorTeamAbbr'].to_list())))+' different VisitorTeamAbbr')
data.groupby(['VisitorTeamAbbr','Yards'])['VisitorTeamAbbr'].count()
data.groupby(['VisitorTeamAbbr','A'])['VisitorTeamAbbr'].count()
data.groupby(['VisitorTeamAbbr','S'])['VisitorTeamAbbr'].count()
data.groupby(['VisitorTeamAbbr','Age'])['VisitorTeamAbbr'].count()
data.groupby(['Age','Yards'])['Age'].count()
print('Highest Player was of:',data['PlayerHeight'].max(),'ft-in')

print('Shortest Player was of:',data['PlayerHeight'].min(),'ft-in')

#print('Average Tall on the field:',data['PlayerHeight'].mean(),'ft-in') # We need to convert 

#this categorical feature to be able to get Average Tall value
data.groupby(['PlayerHeight','Yards'])['PlayerHeight'].count()
JerseyNumber = data['JerseyNumber'].to_list()

len(set(JerseyNumber))
data.groupby(['JerseyNumber','Yards'])['JerseyNumber'].count()
print('Fastest Player was of:',data['S'].max(),'yards/second^2')

print('lowest Player was of:',data['S'].min(),'yards/second^2')

print('Average speed on the field:',data['S'].mean(),'yards/second^2')
data.groupby(['JerseyNumber','S'])['JerseyNumber'].count()
data.groupby(['JerseyNumber','S'])['JerseyNumber'].count()
data.groupby(['JerseyNumber','Age'])['JerseyNumber'].count()
print('The Player with highest Accelararation was of:',data['A'].max(),'yards/second')

print('lowest Player Accelararation was of:',data['A'].min(),'yards/second')

print('Average Accelararation on the field:',data['A'].mean(),'yards/second')
data.groupby(['JerseyNumber','A'])['JerseyNumber'].count()
print('Fat Player was of:',data['PlayerWeight'].max(),'lbs')

print('Skiny Player was of:',data['PlayerWeight'].min(),'lbs')

print('Average Weight on the field:',data['PlayerWeight'].mean(),'lbs')
data.groupby(['JerseyNumber','PlayerWeight'])['JerseyNumber'].count()
sns.factorplot('Yards','Experience',col='Quarter',data=data)

plt.show()
sns.factorplot('Yards','Experience',col='Down',data=data)

plt.show()
sns.factorplot('Yards','Experience',col='PlayDirection',data=data)

plt.show()
print('Highest Player was of:',data['PlayerWeight'].max(),'lbs')

print('Shortest Player was of:',data['PlayerWeight'].min(),'lbs')

print('Average Tall on the field:',data['PlayerWeight'].mean(),'lbs')
print('Highest Player was of:',data[''].max(),'')

print('Shortest Player was of:',data[''].min(),'')

print('Average Tall on the field:',data[''].mean(),'')
print('Highest Player was of:',data[''].max(),'')

print('Shortest Player was of:',data[''].min(),'')

print('Average Tall on the field:',data[''].mean(),'')