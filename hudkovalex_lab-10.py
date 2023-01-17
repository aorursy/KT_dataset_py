import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
football = pd.read_csv('../input/euro-2012new/Euro 2012 stats TEAM.csv')

print(football.head())
print('strings =', football.shape[0]) #1

print('columns =', football.shape[1])
a = football[['Team', 'Yellow Cards', 'Red Cards']] #2

print(a.sort_values(['Red Cards', 'Yellow Cards'], ascending = False))
print(football['Goals'].mean()) #3
print(football['Team'][football['Goals'] >= 6]) #4
football['Difference Of Goals'] = football['Goals'] - football['Goals conceded'] #5

print(football.sort_values(['Difference Of Goals'], ascending = False).head())
print(football.sort_values(['Hit Woodwork'], ascending = False)['Team'].iloc[0]) #6

print(football['Total shots (inc. Blocked)'][football['Team'] == 'Ukraine'].iloc[0])
football['Europe'] = ['other', 'east', 'other', 'west', 'west', 'west', 'other', 'other', 'west', 'east', 'other', 'west', 'east', 'other', 'other', 'east'] #7

print(football.groupby(['Europe'])['Goals'].mean())
a = football.sort_values(['Goals'], ascending = True)[['Team', 'Goals']][:5] #8

print(a)



fig, ax = plt.subplots(figsize = (8, 3))



ax.bar(a['Team'], a['Goals'])



plt.show()
b = football.sort_values(['Corners Taken'], ascending = True)[['Team', 'Corners Taken']] #9



fig, ax = plt.subplots(figsize = (25, 3))



ax.bar(b['Team'], b['Corners Taken'])



plt.show()
c = football.sort_values(['Shots on target', 'Shots off target'], ascending = True)[['Team', 'Shots on target', 'Shots off target']] #10



fig, ax = plt.subplots()



plt.scatter(c['Shots off target'], c['Shots on target'], color = 'green')





plt.show()
