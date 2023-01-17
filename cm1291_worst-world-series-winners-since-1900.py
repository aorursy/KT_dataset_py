import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings



#Ignore warnings

warnings.filterwarnings('ignore')



#Read through team.csv

data = pd.read_csv('../input/team.csv')



#Get the rows needed

d1 = data[['year','team_id','g','w','r','ra','ws_win']]



#We want only WS winners

d2 = d1[d1['ws_win']=='Y']



#Modern era (year > 1900)

d3 = d2[d2['year']>1900]
#Calculate winning %

d3['winpct'] = d3['w']/d3['g']*100



#Adjusted Run Diff

d3['ard'] = (d3['r']/d3['g']*162) - (d3['ra']/d3['g']*162)



#New data

d4 = d3[['year','team_id','ard','winpct']]



sorted = pd.DataFrame.sort_values(d4,'ard')



print(sorted.head(10))
#Plot data

x = d4['ard']

y = d4['winpct']



plt.axis([-30,x.max()+10, 50, y.max()+10])

plt.xlabel('Adjusted Run Differential')

plt.ylabel('Win %')

plt.title("Worst WS team, modern era")



plt.scatter(x,y)



plt.annotate('1987 Twins', xy=(-19,52.7), xycoords='data', xytext=(-20, 63), size=10, arrowprops=dict(arrowstyle="simple",fc="0.6", ec="none"))

plt.annotate('2006 Cardinals', xy=(25,51.552795), xycoords='data', xytext=(100, 52), size=10, arrowprops=dict(arrowstyle="simple",fc="0.6", ec="none"))