# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv") #read data file to start
data.info() #first look for the dataset
data.head(5) # check first 5 rows of the dataset for better information
data.drop(["Unnamed: 0","Flag","Photo","Club Logo"],axis = 1,inplace = True)
data.head()
def groupby_dataset(data,groupby_column,sort_column):
    new_data = data.groupby(groupby_column).median().sort_values(sort_column,ascending = False).head() #select first 5 rows of descending ordered data according to sort column
    return new_data

#Group dataset by something such as Nationality, Club and take median of it with descending order to get highest 5 of grouped by item.
data_nat_overall = groupby_dataset(data,"Nationality","Overall") #get median overall of each Nationality
data_nat_potential = groupby_dataset(data,"Nationality","Potential") #get median potential of each Nationality
data_nat_potential

plt.figure(figsize=(25,8))
plt.subplot(1,2,1)

sns.set(style="whitegrid")
ax = sns.barplot(x=data_nat_overall.index, y=data_nat_overall["Overall"], data=data_nat_overall)
plt.title("Mean Overall of Top 5 Countries")

plt.subplot(1,2,2)
ax = sns.barplot(x=data_nat_potential.index, y=data_nat_potential["Overall"], data=data_nat_potential)
plt.title("Mean Potential of Top 5 Countries")
data_club_overall = groupby_dataset(data,"Club","Overall") #get median overall of each Club
data_club_potential = groupby_dataset(data,"Club","Potential") #get median potential of each Club
data_club_potential
plt.figure(figsize=(25,8))
plt.subplot(1,2,1)

sns.set(style="whitegrid")
ax = sns.barplot(x=data_club_overall.index, y=data_club_overall["Overall"], data=data_club_overall)
plt.title("Mean Overall of Top 5 Clubs")

plt.subplot(1,2,2)
ax = sns.barplot(x=data_club_potential.index, y=data_club_potential["Overall"], data=data_club_potential)
plt.title("Mean Potential of Top 5 Clubs")
data.Overall.hist(figsize = (10,6),bins = 45)
plt.show()
#Distribution looks like a normal distribution with one peak around 63
plt.figure(figsize = (12,8))
sns.scatterplot(x = data.Special, y = data.Overall, data = data)
plt.title("Special vs Overall on Players")
plt.show()
data.columns
data.drop(['ID','LS', 'ST',
       'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM',
       'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB',
       'RCB', 'RB', ],axis = 1, inplace = True)
data.columns
# Drop ID, and position columns since I do not need them

data.info()
attributes = (
    'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes'
)

# Top four attributes per position
for i, val in data.groupby(data['Position'])[attributes].mean().iterrows():
    print("For Position {}, effect of most important attributes on overall is:".format(i))
    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
    a=1
    plt.figure(figsize = (12,12))
    for ind in tuple(val.nlargest(4).index):
        
        plt.subplot(2,2,a)
        data_position = data[data["Position"]== i]
        plt.scatter(data_position[ind],data_position["Overall"])
        plt.title("Attribute {} vs Overall of Position {}".format(ind,i))
        plt.xlabel(ind)
        plt.ylabel("Overall")
        a = a+1
    plt.show()    
