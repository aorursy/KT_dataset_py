import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Params for method Chart
Indicator1 = {'%3sMade': {'y': '%3sMade', 'hue': None, 'hue_order': None, 'yaxis': 'The Percent of field goals that was a 3PT Field Goal', 'title': 'The Percentage a team Shoots and makes a 3 Pointer', 'legend': False}, '%3sAttempted': {'y': '%3sAttempted', 'hue': None, 'hue_order': None, 'yaxis': 'The Percent of field goals that was a 3PT Field Goal', 'title': 'The Percentage a team Shoots a 3 Pointer', 'legend': False}}
Indicator2 = {'SHOTS_ATTEMPTED by LOCATION': {'y': 'SHOTS_ATTEMPTED', 'hue': 'LOCATION', 'hue_order': ['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'], 'yaxis': 'Amount of Shots Attempted', 'title': 'The amount of shots attempted for different locations', 'legend': True}, 'FG% by LOCATION': {'y': 'FG%', 'hue': 'LOCATION', 'hue_order': ['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'], 'yaxis': 'Field Goal Percentage', 'title': 'The FG% for different locations', 'legend': True}, 'eFG% by LOCATION': {'y': 'eFG%', 'hue': 'LOCATION', 'hue_order': ['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'], 'yaxis': 'Effective Field Goal Percentage', 'title': 'The eFG% for different locations', 'legend': True}}
Indicator3 = {'Bucket List': {'y': 'SHOTS_ATTEMPTED', 'hue': 'SHOT_DISTANCE', 'hue_order': range(40), 'yaxis': 'Amount of Shots Attempted', 'title': 'The amount of shots attempted for every distance', 'legend': True}, 'SHOTS_ATTEMPTED by SHOT_DISTANCE': {'y': 'SHOTS_ATTEMPTED', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Amount of Shots Attempted', 'title': 'The amount of shots attempted for different distances', 'legend': True}, 'SHOTS_MADE by SHOT_DISTANCE': {'y': 'SHOTS_MADE', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Amount of Shots Made', 'title': 'The amount of shots made for different distances', 'legend': True}, 'FG% by SHOT_DISTANCE': {'y': 'FG%', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Field Goal Percentage', 'title': 'The FG% for different distances', 'legend': True}, 'eFG% by SHOT_DISTANCE': {'y': 'eFG%', 'hue': 'SHOT_DISTANCE2', 'hue_order': None, 'yaxis': 'Effective Field Goal Percentage', 'title': 'The Effective FG% for different distances', 'legend': True}}
Indicator4 = {'SHOTS_MADE by PERIOD': {'y': 'eFG%', 'hue': 'PERIOD', 'hue_order': [1,2,3,4], 'yaxis': 'Effective Field Goal Percentage', 'title': 'The Effective FG% for different Periods/Quarters', 'legend': True}, 'eFG% by MINUTES_REMAINING': {'y': 'eFG%', 'hue': 'MINUTES_REMAINING', 'hue_order': [11,10,9,8,7,6,5,4,3,2,1,0], 'yaxis': 'Effective Field Goal Percentage', 'title': 'The Effective FG% for every Minute remaining in the Quarter', 'legend': True}}
Indicator = [None, Indicator1, Indicator2, Indicator3, Indicator4]

sns.set(style="darkgrid")

#Method that build chart based on parameters 
def Chart(indicator, chart, ncol=1, x=0.8):
	data = pd.read_csv('../input/Indicator' + str(indicator) + '.csv')
	ax = sns.barplot(x='TEAM_RATING', y=Indicator[indicator][chart]['y'], hue=Indicator[indicator][chart]['hue'], data=data, order=['Good','Avg','Bad'], hue_order=Indicator[indicator][chart]['hue_order'], ci=None)
	ax.set(xlabel='Team Rating', ylabel=Indicator[indicator][chart]['yaxis'])
	ax.set_title(Indicator[indicator][chart]['title'])
	if Indicator[indicator][chart]['legend']:
		plt.legend(loc=(1,x), ncol=ncol)
	plt.show()

    


plt.figure(figsize=(10,5))
Chart(1, '%3sAttempted')
plt.figure(figsize=(10,5))
Chart(1, "%3sMade")

plt.figure(figsize=(15,8))
Chart(2,'SHOTS_ATTEMPTED by LOCATION')
plt.figure(figsize=(15,8))
Chart(2,'FG% by LOCATION')
plt.figure(figsize=(15,8))
Chart(2,'eFG% by LOCATION')
#Remove the warriors and rockets
data = pd.read_csv('../input/Indicator2.csv')
data = data[data['TEAM_NAME'] != 'Houston Rockets'] 
data = data[data['TEAM_NAME'] != 'Golden State Warriors']

plt.figure(figsize=(15,8))
ax = sns.barplot(x='TEAM_RATING', y='SHOTS_ATTEMPTED', hue='LOCATION',data=data, order=["Good", "Avg", "Bad"], hue_order=['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'])
ax.set(xlabel='Team Rating', ylabel='Amount of Shots Attempted')
ax.set_title('The amount of shots attempted for different locations w/out the Rockets and Warriors')	
plt.legend(loc=(1,0.8))
plt.show()
    
plt.figure(figsize=(15,8))
ax = sns.barplot(x='TEAM_RATING', y='FG%', hue='LOCATION',data=data, order=["Good", "Avg", "Bad"], hue_order=['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'])
ax.set(xlabel='Team Rating', ylabel='Field Goal Percentage')
ax.set_title('The FG% for different locations w/out the Rockets and Warriors')
plt.legend(loc=(1,0.8))
plt.show()

plt.figure(figsize=(15,8))  
ax = sns.barplot(x='TEAM_RATING', y='eFG%', hue='LOCATION',data=data, order=["Good", "Avg", "Bad"], hue_order=['Left Corner 3','Above the Break 3','Right Corner 3','Mid-Range','In The Paint (Non-RA)','Restricted Area'])
ax.set(xlabel='Team Rating', ylabel='Effective Field Goal Percentage')
ax.set_title('The eFG% for different locations w/out the Rockets and Warriors')
plt.legend(loc=(1,0.8))
plt.show()
plt.figure(figsize=(16,8))
Chart(3,'Bucket List',3, 0.52)

plt.figure(figsize=(14,6))
Chart(3,'SHOTS_ATTEMPTED by SHOT_DISTANCE')
plt.figure(figsize=(14,6))
Chart(3, 'SHOTS_MADE by SHOT_DISTANCE')
plt.figure(figsize=(14,6))
Chart(3, 'FG% by SHOT_DISTANCE')
plt.figure(figsize=(14,6))
Chart(3, 'eFG% by SHOT_DISTANCE')
plt.figure(figsize=(16,8))
Chart(4, 'SHOTS_MADE by PERIOD')
plt.figure(figsize=(16,8))
Chart(4, 'eFG% by MINUTES_REMAINING')
#Good Teams 
data = pd.read_csv('../input/Indicator4.csv')
data = data[data['TEAM_RATING'] != 'Bad']
data = data[data['TEAM_RATING'] != 'Avg'] 
plt.figure(figsize=(16,8))
ax = sns.barplot(x='TEAM_NAME', y='eFG%', hue='MINUTES_REMAINING',data=data, hue_order=[11,10,9,8,7,6,5,4,3,2,1,0], ci=None)
ax.set(xlabel='Team Name', ylabel='Effective Field Goal Percentage')
ax.set_title('The Effective FG% for every Minute remaining in the quarter for the good teams')
plt.legend(loc=(1,0))
plt.show()
 
#Bad Teams
data = pd.read_csv('../input/Indicator4.csv')
data = data[data['TEAM_RATING'] != 'Good']
data = data[data['TEAM_RATING'] != 'Avg'] 
plt.figure(figsize=(16,8))
ax = sns.barplot(x='TEAM_NAME', y='eFG%', hue='MINUTES_REMAINING',data=data, hue_order=[11,10,9,8,7,6,5,4,3,2,1,0], ci=None)
ax.set(xlabel='Team Name', ylabel='Effective Field Goal Percentage')
ax.set_title('The Effective FG% for every Minute remaining in the quarter for the bad teams')
plt.legend(loc=(1,0))
plt.show()