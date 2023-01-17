#Importing libraries

import pandas as pd

import numpy as np

import scipy as sci

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

import math
dataset = pd.read_csv('../input/austin_crime.csv')
dataset.shape
#In order to learn missing data

msno.bar(dataset)
#Extracting empty rows with respect to x_coordinate

dataset = dataset[pd.notnull(dataset['x_coordinate'])]

dataset = dataset.reset_index()
x_border = (dataset.x_coordinate.min(), dataset.x_coordinate.max())

y_border = (dataset.y_coordinate.min(), dataset.y_coordinate.max())
#Converting the districs

numerical={'A':'red','B':'green','C':'blue','D':'black','E':'purple','F':'orange','G':'cyan','H':'gray','I':'yellow','AP':'brown','UK':'pink'}

dataset['district_colored']=dataset['district'].map(numerical)
fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True)

ax.scatter(dataset['x_coordinate'].values, dataset['y_coordinate'].values,

             s=3, c=dataset.district_colored.values , label='map', alpha=0.5)

fig.suptitle('Crime Places with respect to x_coordinate and y_coordinate')

ax.legend(loc=1)

ax.set_ylabel('y_coordinate')

ax.set_xlabel('x_coordinate')

plt.ylim(y_border)

plt.xlim(x_border)

plt.show()
temp= dataset[pd.notnull(dataset['latitude'])]



long_border = (dataset.longitude.min(), dataset.longitude.max())

lat_border = (dataset.latitude.min(), dataset.latitude.max())
fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True)

ax.scatter(temp['longitude'].values, temp['latitude'].values,

             s=3, c=temp.district_colored.values , label='map', alpha=0.5)

fig.suptitle('Crime Places with respect to Latitude and Longitude')

ax.legend(loc=0)

ax.set_ylabel('latitude')

ax.set_xlabel('longitude')

plt.ylim(lat_border)

plt.xlim(long_border)

plt.show()
#X_coordinate&longitude ; y_coordinate&latitude look like each other .
#District

district_values =pd.value_counts(dataset['district'])

district_values.plot(kind="bar",color='yellow')

_=plt.xlabel('District')

_=plt.ylabel('How many crimes?')

plt.show()
#Most of crime occured at the regions of D , E , F
#MURDER

#I will focus on MURDER.



dataset['murder'] = np.zeros(len(dataset))



for i in range(0,72273):

    if dataset.loc[i,'description'] == 'MURDER' :

        dataset.loc[i,'murder'] = 1



numerical2={1:'red',0:'yellow'}

dataset['murder_colored']=dataset['murder'].map(numerical2)
dataset.shape
fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True)

ax.scatter(dataset['x_coordinate'].values, dataset['y_coordinate'].values,

             s = 1 + pow(30,dataset.murder) , c=dataset.murder_colored, label='map', alpha=0.5)

fig.suptitle('Murder map')

ax.legend(loc=1)

ax.set_ylabel('y_coordinate')

ax.set_xlabel('x_coordinate')

plt.ylim(y_border)

plt.xlim(x_border)

plt.show()
#It can be deducted that Murders are accumulating at southeast of Austin.
#Weekday and month analysis

#week is the order of a week in a year from 1 to 53 .

dataset['clearance_date'] = pd.to_datetime(dataset.clearance_date)



dataset['month'] = dataset['clearance_date'].dt.month

dataset['day'] = dataset['clearance_date'].dt.day

dataset['week'] = dataset['clearance_date'].dt.week
dataset['week_day_numerical'] = dataset['clearance_date'].dt.weekday



conversion={0:'monday',1:'tuesday',2:'wednesday',3:'thursday',4:'friday',5:'saturday',6:'sunday'}



dataset['week_day'] = dataset['week_day_numerical'].map(conversion)
#Yearly and monthly analysis together

dataset = dataset[pd.notnull(dataset['month'])]
a=dataset.groupby(['month','year']).size()

b=a.unstack(level=-1)

e=b.astype(int)
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5]) 

ax0 = plt.subplot(gs[0])

sns.heatmap(e,annot=True,fmt='d',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":10})

ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(e.sum(axis=1)),annot=True,fmt='d',linewidths=.5,ax=ax1,cbar=False,annot_kws={"size":10})

plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Month",size=18)

ax0.set_xlabel("Year",size=18)

ax1.set_xticklabels(["Total"],size=16)

ax0.set_title("Year vs Month ",size=22,y=1.05,x=0.5)

#July is the most dangerous month.  November is the calmest month .
##Weekday analysis

weekday_values =pd.value_counts(dataset['week_day'])

weekday_values.plot(kind="bar",color='red')

_=plt.xlabel('Weekday')

_=plt.ylabel('How many crimes?')

plt.show()
#This is an outstanding result. Most of the crimes occurred in working days ,

# especially tuesday and wednesday .
##Week no

plt.plot(dataset[dataset['year'] == 2014].groupby('week').count()[['district']], 'o-', label='2014')

plt.plot(dataset[dataset['year'] == 2015].groupby('week').count()[['district']], 'o-', label='2015')

plt.title('2014 and 2015 period complete overlap.')

plt.legend(loc=0)

plt.ylabel('number of crimes')

plt.show()
##Description- PIE chart

plt.figure(figsize=(12,8))

dataset.description.value_counts(sort=True).head(20).plot(kind='pie',autopct='%1.1f%%')

plt.title('Number of Descriptions')

plt.show()
#As can be seen, most of the crimes are related to burglary, shoplifting, thief and robbery
## Week Evaluation , 2014



plt.figure(figsize=(14,10))

sns.despine()

sns.distplot(dataset[dataset['year'] == 2014]['week'],kde_kws={"color":"g","lw":4,"label":"KDE Estimation","alpha":0.5},

            hist_kws={"color":"r","alpha":0.5,"label":"Frequency"});

plt.xlim(0,54)

plt.xticks(np.arange(1,54),size=7)

plt.yticks(size=14)

plt.ylabel("Density",rotation=90, size=20)

plt.xlabel("Week of Year",size=20)

plt.show() 
# One may think some police are cooperating with theives adn robbers .

# However, in some weeks, the number of crimes are doubled.

# The pattern is in this way: 2 weeks high, 1 week low, 1 week high ,1 week low.

# It repeats in this way .  
#Week Evaluation , 2015

plt.figure(figsize=(14,10))

sns.despine()

sns.distplot(dataset[dataset['year'] == 2015]['week'],kde_kws={"color":"g","lw":4,"label":"KDE Estimation","alpha":0.5},

            hist_kws={"color":"r","alpha":0.5,"label":"Frequency"});

plt.xlim(0,54)

plt.xticks(np.arange(1,54),size=7)

plt.yticks(size=14)

plt.ylabel("Density",rotation=90, size=20)

plt.xlabel("Week of Year",size=20)

plt.show()
#Peaks are shown in a remarkable way in the pattern of one week high, one week low
#CONCLUSION

#It is highly possible for some police to act together with thives and robbers

#Most of the crime is related to theft and robbery .

#Crimes occur in weekdays most probably