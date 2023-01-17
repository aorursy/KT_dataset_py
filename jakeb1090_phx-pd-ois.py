# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

#%matplotlib inline

df = pd.read_csv('../input/officer-involved-shooting-incidents_officer-involved-shooting-incidents-2017-forward_ois1.csv')

from datetime import datetime

#datetime.strptime('04-APR-18','%d,%b,%y')

df.sample(7)
import pandas as pd

import pandas_profiling

profile = df.profile_report(title='Pandas Profiling Report')

profile.to_file(output_file="fifa_pandas_profiling.html")



pandas_profiling.ProfileReport(df)
council_dist = df.groupby(['COUNCIL_DIST']).size().sort_values(ascending=False)

council_dist.plot(kind='bar', color='teal', edgecolor='aqua', title='Shootings by City Council District\n\n')

plt.tick_params(top='off', bottom='False', left='off', right='off', labelbottom='on')

#Result from shooting incident

result = df.groupby(['SP_INJURY_LEVEL']).size()

colors = ['gray','teal', 'cyan', 'turquoise', 'deepskyblue', 'lightseagreen']

result.plot(kind='pie', colors=colors, label=True)

title='jej'

centre_circle = plt.Circle((0,0),0.7,fc='white')

fig = plt.gcf()



fig.gca().add_artist(centre_circle)
sp_race = df.groupby(['SP_ETHNICITY']).size()

sp_race.plot(kind='pie', startangle=-90, title='Suspects by Race', colors=colors)

officers_numbers = len(df)

print(officers_numbers)

plt.tick_params(top=False, bottom=False, left=False, labelleft=False, right=False, labelbottom=True)

po_race = df.groupby(['PO_RACE']).size()

sns.set(style='whitegrid')

po_race.plot(kind='barh', title='Officer by Race',width=.847, color='c', edgecolor='cyan')

#

plt.tick_params(top=False, bottom=False, left=False, right=False, grid_visible=False)



#sns.countplot(data=df['SP_GENDER'], hue='gender')
by_precinct = df.groupby(['PRECINCT']).size().sort_values(ascending=True)

by_precinct.plot(kind='barh', 

                 color='gray', 

                 width=.9911,

                 title='Shootings by Precinct', 

                 edgecolor='cyan',

                )

precinct = pd.DataFrame(by_precinct)

precinct



#plt.figure(figsize=())

#time of day

tod = df.groupby(['HOUR']).size()

tod.plot(kind='bar', color='white', 

         edgecolor='teal', width=1,

        title='Shooting Incidents by Time of Day', grid=False

        )

for spine in plt.gca().spines.values():

    spine.set_visible(False)

print(len(df.HOUR))



plt.tick_params(top=False, bottom=False, right=False)



#plt.figure(figsize=(6,4))

#sns.countplot(x=df['HOUR'])
#day of week

dow = df.groupby(['DAY_OF_WEEK']).size()

#dow.plot(kind='barh')

dow.plot(kind='bar', color='lightseagreen', edgecolor='cyan',

         width=1, title='Shootings by DOW' , grid=False)



for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

age_group = df.groupby(['SP_AGE_GROUP']).size()

#sns.set(style='whitegrid')

age_group.plot(kind='pie', title='Suspects by Age', colors=colors )

df.head()
#geo = open('https://opendata.arcgis.com/datasets/efd9cb91283e4aec906a79cf022a6988_0.geojson')

#race and age

plt.figure(figsize=(18,14))

sns.countplot(y=df['LOCATION'],

              order=df['LOCATION'].value_counts().index,

              

         )
woc = df['SP_WEAPON'].groupby(df['SP_INJURY_LEVEL'])

sns.countplot(y=df['SP_WEAPON'])
sns.countplot(y=df['LOCATION'])
