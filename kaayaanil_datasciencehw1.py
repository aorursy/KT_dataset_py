# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting

import seaborn as sns #data visualization

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#creating an dataframe

df = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv') #read_csv is reading csv files

df.head(10) #lookin first 10 index in dataframe
df.isnull().sum()
df.columns.tolist() #all columns in data
df2 = df[['short_name','age','height_cm','weight_kg','nationality','club', #new data frame consists major features

 'overall','potential','value_eur','wage_eur',

 'player_positions',

 'preferred_foot',

 'international_reputation',

 'weak_foot',

 'skill_moves',

 'work_rate',

 'body_type',

 'real_face',

 'release_clause_eur',

 'player_tags',

 'team_position',

 'team_jersey_number',

 'loaned_from',

 'joined',

 'contract_valid_until',

 'nation_position',

 'nation_jersey_number',

 'pace',

 'shooting',

 'passing',

 'dribbling',

 'defending',

 'physic',

 'player_traits',

 'attacking_crossing',

 'attacking_finishing',

 'attacking_heading_accuracy',

 'attacking_short_passing',

 'attacking_volleys',

 'skill_dribbling',

 'skill_curve',

 'skill_fk_accuracy',

 'skill_long_passing',

 'skill_ball_control',

 'movement_acceleration',

 'movement_sprint_speed',

 'movement_agility',

 'movement_reactions',

 'movement_balance',

 'power_shot_power',

 'power_jumping',

 'power_stamina',

 'power_strength',

 'power_long_shots',

 'mentality_aggression',

 'mentality_interceptions',

 'mentality_positioning',

 'mentality_vision',

 'mentality_penalties',

 'mentality_composure',

 'defending_marking',

 'defending_standing_tackle',

 'defending_sliding_tackle']]
df2.head()
df2.info() #looking some informations about data frame
df2.describe() #show some informations about numeric columns
#Correlation map

f,ax = plt.subplots(figsize = (25,16))

sns.heatmap(df2.corr(),annot = True,linewidths=.5,fmt = '.1f',ax=ax,cmap='YlGnBu')

plt.title('Heat Map')

plt.show()
#Line plot

df2.overall.plot(kind = 'line',color = 'red',label = 'overall',lw = 1,alpha = 0.5, grid = True,figsize = (13,13))

df2.dribbling.plot(color = 'green',label='age',lw = 1,alpha = 0.5,grid=True,ls = '-.')

plt.legend(loc = 'upper right')

plt.xlabel = ('x axis')

plt.ylabel = ('y axis')

plt.title('line plot')

plt.show()
#Scatter Plot compere to columns



df2.plot(kind = 'scatter',x = 'pace',y = 'shooting',color = 'red',alpha = 0.3,grid = True,figsize = (8,4.5))

plt.ylabel = 'shoot'

plt.xlabel = 'pace'

df2.plot(kind = 'scatter',x = 'physic',y = 'shooting',color = 'red',alpha = 0.3,grid = True,figsize = (8,4.5))

plt.ylabel = 'shoot'

plt.xlabel = 'pace'



plt.show()
df2.plot(kind = 'scatter', x ='passing',y = 'skill_curve',color = 'green',grid = True,figsize = (16,9))

plt.xlabel = 'Passing'

plt.ylabel = 'Skill Curve'

plt.show()
#age Histogram

df2.age.plot(kind = 'hist',bins = 80,figsize = (16,9))

plt.show()
dic = {'spain':'Real Madrid','england':'Liverpool','italy':'Juventus','holland':'Ajax'}

print(dic.keys())

print(dic.values())
print(dic['spain'])

dic['spain'] = 'barcelona'

print(dic)
dic['germany'] = 'Dortmund'

print(dic)
del dic['holland'] #deleting key and value in list

print(dic)
print(dic.items())

print('germany' in dic) #boolean result

#pd.DataFrame(dic) Creating data frame 

dic.clear() #deleting all items in dictionary

print(dic)

#del dic deleting dic variable in memory
df2
series = df2['potential']

print(type(series))

dataframee = df2[['potential']]

print(type(dataframee))
#filtering datas

filt = df2['overall'] > 89

df3 = df2[filt]

df3
df2[(df2['power_long_shots']>80) & (df2['overall']>90)] #to merge two diffrent filter
df2[np.logical_and(df2['overall']>90,df2['power_long_shots']>80)] #to merge with np logical and
#for loop in data

for index,value in df2[['overall']][0:10].iterrows():

    print(index,': ',value) 