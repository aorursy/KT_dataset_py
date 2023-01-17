# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
#First we look datas info

data.info()
#After we writing datas corr

data.corr()
#Let's look at the top five to examine in more detail

data.head()
#After we learn columns

data.columns
#Let's delete columns that we don't need

list2=['Photo','Flag','Club Logo', 'Special',

       'Preferred Foot', 'International Reputation', 'Weak Foot',

       'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Joined', 'Loaned From',

       'Contract Valid Until','Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 

       'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

for i in list2:

    del data[i]

data.head()
#Filtering

data[(data['Overall']>=85)]
#Line Plot

data.Overall.plot(kind = 'line', color = 'b',label = 'Overall',linewidth=1,alpha = 1,grid = True,linestyle = '-')

plt.legend(loc='upper right')

plt.xlabel('Footballers')

plt.ylabel('Overall')

plt.title('Line Plot')

plt.show()
# Scatter Plot 

data.plot(kind='scatter', x='Overall', y='Age',alpha = 1,color = 'orange')

plt.xlabel('Overall')

plt.ylabel('Age')

plt.title('Overall / Age Scatter Plot')

plt.show()
# Histogram

data.Overall.plot(kind = 'hist',bins = 100,figsize = (12,12))

plt.show()
# clf()

data.Overall.plot(kind = 'hist',bins = 50)

plt.clf()
dictionary = {'L. Messi' : 94,'Cristiano Ronaldo' : 94}

print(dictionary.keys())

print(dictionary.values())
dictionary['Cristiano Ronaldo'] = 95                     #{'L. Messi': 94, 'Cristiano Ronaldo': 95}

print(dictionary)

dictionary['Neymar Jr'] = 93                             #{'L. Messi': 94, 'Cristiano Ronaldo': 95, 'Neymar Jr': 93}

print(dictionary)

del dictionary['Neymar Jr']                              #{'L. Messi': 94, 'Cristiano Ronaldo': 95}

print(dictionary)

print('Neymar Jr' in dictionary)                         #False

dictionary.clear()

print(dictionary)                                        #{}
x = data['Overall']>=85                                  #We have "Overall>= 85" 110 footballers

data[x]
# 2 Filtering

data[(data['Overall']>90) & (data['Age']<35)]           #1 lbs just about 0.45 kg 190 lbs just about 85,5 kg
#while loop

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
#for loop

for index,value in data[['Overall']][0:5].iterrows():

    print(index," : ",value)
#How many footballers are there? 

def howmany(x):

    return len(x)

howmany(data['Unnamed: 0'])

#OR

def howmanyitems(*args):

    print(len(*args))

howmanyitems(data['Unnamed: 0'])
print(data['Position'].value_counts(dropna =False))
data.describe()