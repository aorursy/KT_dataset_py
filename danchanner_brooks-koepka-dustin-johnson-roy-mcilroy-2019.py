import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Load the PGA data

f = '/kaggle/input/pga-tour-20102018-data/2019_data.csv'

data = pd.read_csv(f)

data.head()
#Stats & players we are interested in

s1 = 'Driving Distance - (AVG.)'

s2 = 'Hit Fairway Percentage - (%)'

s3 = 'Greens in Regulation Percentage - (%)'

s4 = "Putting from - > 10' - (% MADE)"

s5 = "Putting from 4-8' - (% MADE)"

s6 = "Scrambling - (%)"

s7 = "Sand Save Percentage - (%)"

stats = [s1,s2,s3,s4,s5,s6,s7]

p1 = 'Brooks Koepka'

p2 = 'Rory McIlroy'

p3 = 'Dustin Johnson'

players = [p1,p2,p3]



#filter the dataset down to the key stats and players

data.describe()

df = data.loc[data['Variable'].isin(stats)]

df = df.loc[df['Player Name'].isin(players)]

df.head()
df['Value'] = pd.to_numeric(df["Value"])

df.dtypes
#List to hold stat, shorter description and units

a = [[s1, 'Driving Distance','Yards'],

    [s2,'Fairways Hit',"(%)"],

    [s3,'Greens in Regulation',"(%)"],

    [s4,'Putts made outside 10ft',"(%)"],

    [s5,'Putts made 4 to 8 ft',"(%)"],

    [s6,'Scrambling',"(%)"],

    [s7,'Sand Saves',"(%)"]]
#Plot the data

for n in range(7):

    d = df[df['Variable']==a[n][0]]

    for p in players:

        sns.distplot(d['Value'][d['Player Name']==p],bins=50, hist=False,label=p)

    plt.title(a[n][1])

    plt.xlabel(a[n][2])

    plt.show()