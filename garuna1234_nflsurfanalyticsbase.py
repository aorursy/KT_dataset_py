# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#READING all the csv downloaded from Kaggle API

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.cm as cmap

import seaborn as sns 
playerlist = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

injury_record = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

trackerdata = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
relationship = playerlist.drop_duplicates('PlayerKey', keep='first').merge(injury_record, on='PlayerKey')

relationship['RosterPosition'].value_counts().plot(kind='bar', title='INJURIES VS Position of Player')
df = relationship[['RosterPosition','Surface','BodyPart']]

dfnew = df.groupby(['RosterPosition', 'BodyPart']).count()

#

#result = df.pivot(index='Surface', columns='RosterPosition', values='value')

dfrev = dfnew['Surface'].reset_index().rename(columns={'Surface':'Count'})

ax = sns.heatmap(dfrev.pivot_table(index='RosterPosition', columns = 'BodyPart', values='Count'))
surfaceinjuries = relationship[['RosterPosition', 'Surface']].groupby(['RosterPosition', 'Surface']).size().reindex()

lst = []

lsttwo = []

counter = 0

countertwo = 1

#Natural Surface Injuries  

while counter < len(surfaceinjuries):

    lst.append(int(surfaceinjuries[counter]))

    counter = counter + 2

    

    

#Synthetic Surface Injuries

while countertwo < len(surfaceinjuries)+1:

    lsttwo.append(surfaceinjuries[countertwo])

    countertwo = countertwo + 2

    

# set width of bar

barWidth = 0.25

 

# set height of bar

bars1 = lst

bars2 = lsttwo

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

 

# Make the plot

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Natural')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Synthetic')

#plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

 

# Add xticks on the middle of the group bars

plt.xlabel('Position', fontweight='bold')

plt.ylabel('Injuries', fontweight='bold')



plt.xticks([r + barWidth for r in range(len(bars1))], ['Cornerback', 'Defensive Lineman', 'Linebacker', 'Offensive Lineman', 'Running Back', 'Safety', 'Tight End', 'Wide Receiver'])



plt.xticks(rotation='vertical')

plt.title('NFL Surface Lower-Limb Injuries', fontweight='bold')

# Create legend & Show graphic

plt.legend()

plt.show()





#Creating Stacked Bar with SurfaceInjuries among Lower-Limb NFL candidates

#relationship[['RosterPosition', 'Surface']].groupby(['RosterPosition', 'Surface'])

    

#for x in range(0, len(surfaceinjuries), 2):

    #natural = lst.append(surfaceinjuries[x])

#list(surfaceinjuries.to_frame().columns)

numplays = trackerdata.groupby('PlayKey').size().to_frame().reset_index()

injury_record

merged_set = pd.merge(injury_record, numplays, on='PlayKey')

#For 77 NFL lower-limb injures, looking at any possible trends or other signficant factors that highly affect the outcome

#of the total # of plays and number of days missed during practices

merged_set['sum'] = merged_set.iloc[:,5:9].sum(axis = 1)

missed_time = []

for x in merged_set['sum']:

    if x == 4:

        missed_time.append('DM_42')

    elif x == 3:

        missed_time.append('DM_28')

    elif x == 2:

        missed_time.append('DM_7')

    else:

        missed_time.append('DM_1')

merged_set['NumDaysMissed'] = missed_time

missed_time
merged_set = merged_set.iloc[:, [9, 11]].groupby('NumDaysMissed').mean()









merged_set.reindex().rename(columns={'0':'AverageNumPlays'}).plot(kind='bar', title='Average Plays and Relationship with Num Days Missed for Injured NFL Players', legend=None)

plt.xlabel('NumDaysMissed')

plt.ylabel('AveragePlays')

numplays = trackerdata.groupby('PlayKey').size().to_frame().reset_index()

trackerdata
playerlist
playerlist['PlayType']
injury_record
playerlist.loc[playerlist['PlayerKey']==26624]
playerlist.iloc[:, [ 6, 7]].groupby(['StadiumType']).count().reset_index().sort_values(by='FieldType', ascending=False).iloc[0:5].set_index('StadiumType').plot.bar(y='FieldType', rot=0)
