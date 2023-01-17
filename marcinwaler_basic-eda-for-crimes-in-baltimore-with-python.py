import pandas as pd
crime = pd.read_csv(r'../input/BPD_Part_1_Victim_Based_Crime_Data.csv')
crime.shape
crime.info()
#NaN values for each column
crime.isnull().sum(axis = 0)
crime['Weapon'].unique()
crime['Weapon'].fillna('NO WEAPON', inplace=True)
crime.Weapon.unique()
crime.Description.unique()
#Replace dirty data
crime['CrimeTime'] = crime['CrimeTime'].str.replace('24:00:00', '00:00:00')
#Merge date with time and next create new column with datetime type.
crime['Date'] = crime['CrimeDate'] + ' ' +  crime['CrimeTime']
crime['Date'] = pd.to_datetime(crime['Date'])
#Create new columns for further analysis
crime['Day'] = crime['Date'].dt.day
crime['Month'] = crime['Date'].dt.month
crime['Year'] = crime['Date'].dt.year
crime['Weekday'] = crime['Date'].dt.weekday + 1
crime['Hour'] = crime['Date'].dt.hour

#drop columns
crime = crime.drop(['CrimeDate', 'CrimeTime'], axis=1)

#Set datetime index
crime = crime.set_index('Date')
crime['Inside/Outside'].unique()
#Text unification
crime['Inside/Outside'] = crime['Inside/Outside'].replace('I', 'Inside')
crime['Inside/Outside'] = crime['Inside/Outside'].replace('O', 'Outside')
crime['Inside/Outside'].unique()
crime.head()
#Value counts for entire dataframe
for x in crime.columns:
    print(crime[x].value_counts())
    print('='*50)
crime.dtypes
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style(style='darkgrid')
plt.figure(figsize=(8,5))

plt.title('Kind of weapon used in crime')
sns.countplot(x='Weapon', data=crime, order = crime['Weapon'].value_counts().index)
plt.xlabel('Kind of weapon')
plt.ylabel('Number of incidents')
plt.plot()
plt.figure(figsize=(10,5))
plt.title('Kind of weapon used in crime divided into location inside or outside')
sns.countplot(x='Inside/Outside', hue='Weapon', data=crime)
plt.xlabel('Location')
plt.ylabel('Number of incidents')
plt.plot()
plt.figure(figsize=(15,7))

plt.title('The most common crimes in years 2012-2017')
sns.countplot(x='Description', data=crime, hue='Inside/Outside', order = crime['Description'].value_counts().index, )
plt.xticks(rotation=40)
plt.plot()
#Create year and quarter column for next chart
crime['Date'] = crime.index
crime['Quarter'] = crime['Date'].dt.quarter
crime[["Quarter", "Year"]] = crime[["Quarter", "Year"]].astype(str) 
crime['YearQt'] = crime['Year'] + 'Q' + crime['Quarter']
crime.head()
#Group data by year quarter and district
description_agg = crime.groupby(['YearQt', 'District'])['Total Incidents'].sum()
description_agg = description_agg.reset_index(level=[0,1]) #to go from multi index to single index

#select everything instead of Q3 of 2017 year
mask = description_agg['YearQt'] != '2017Q3'
description_agg = description_agg[mask]
plt.figure(figsize=(15,7), dpi=100)
plt.title('Number of incidents for each district in quarters of years 2012-2017', fontsize=16)
sns.lineplot(x='YearQt', y='Total Incidents', hue='District', data=description_agg)
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.ylabel("Number of incidents", fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xticks(rotation=45)
plt.plot()
plt.figure(figsize=(7,4), dpi=80)

plt.title('Number of incidents grouped by weekday of years 2012-2017', fontsize=13)
ax = sns.countplot(x='Weekday', data=crime, color='#5572dd')
plt.ylabel("Number of incidents", fontsize=13)
plt.xlabel('Weekday', fontsize=13)
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',' Saturday', 'Sunday']
ax.set_xticklabels(labels)
plt.plot()
plt.figure(figsize=(7,4), dpi=80)

plt.title('Number of incidents for each year', fontsize=13)
ax = sns.countplot(x='Year', data=crime, color='#5572dd')
plt.ylabel("Number of incidents", fontsize=13)
plt.xlabel('Year', fontsize=13)
plt.plot()
#Let's make number into monthname in crime dataframe
import calendar
crime['Month'] = crime['Month'].apply(lambda x: calendar.month_name[x])


plt.figure(figsize=(7,4), dpi=80)

plt.title('Number of incidents grouped by month of years 2012-2017', fontsize=13)
ax = sns.countplot(x='Month', data=crime, hue='Inside/Outside')
plt.ylabel("Number of incidents", fontsize=13)
plt.xlabel('Month', fontsize=13)
plt.xticks(rotation=45)
plt.plot()
#Let's make secondary dataframe with month name, incidents count and percentage change
months_agg = crime.groupby(['Month'])['Total Incidents'].count()
months_agg = months_agg.reset_index()
months_agg['Pct'] = months_agg['Total Incidents']/months_agg['Total Incidents'].sum()*100

#Month number into month name
import calendar
#months_agg['Month'] = months_agg['Month'].apply(lambda x: calendar.month_name[x])

months_agg.sort_values(by = 'Total Incidents', ascending=False)
from matplotlib.collections import QuadMesh
from matplotlib.text import Text
import numpy as np

#Creating columns for hour of weekday
#weekdaymatrix = pd.DataFrame()
#weekdaymatrix['Weekday'] = pd.DatetimeIndex(crime.index).weekday

#Manipulating data to feed pivot table to then feed to seaborn heatmap
incidents_wh = crime.groupby(['Weekday', 'Hour'])['Total Incidents'].sum()
incidents_wh = incidents_wh.reset_index(level=[0,1]) # to go from mutlindex to singleidnex
pivoted_table = incidents_wh.pivot(index='Hour', columns='Weekday', values='Total Incidents')
pivoted_table.fillna(0, inplace=True)

#Select max value from the data
max_value = pivoted_table.max().max()

#Create sum for rows and columns
pivoted_table.loc['Total'] = pivoted_table.sum()
pivoted_table = pd.concat([pivoted_table,pd.DataFrame(pivoted_table.sum(axis=1),columns=['Total'])],axis=1)

#Generate heatmap
plt.figure(figsize=(18, 8), dpi=90)
ax = sns.heatmap(pivoted_table, cmap='Reds', annot=True, fmt='g', annot_kws={'size': 9}, vmax=max_value)

#==================================Graphical customization code=======================================
#Set white color to total column and row
# find your QuadMesh object and get array of colors
quadmesh = ax.findobj(QuadMesh)[0]
facecolors = quadmesh.get_facecolors()

# make colors of the last column white
column_number = pivoted_table.shape[1]
cells_number = pivoted_table.shape[0]*pivoted_table.shape[1]
last_row = pivoted_table.shape[1]*(pivoted_table.shape[0]-1)

facecolors[np.arange(column_number-1,cells_number,column_number)] = np.array([1,1,1,1]) #change column total to white
facecolors[np.arange(last_row, cells_number,1)] = np.array([1,1,1,1]) #change row total to white

# set modified colors
quadmesh.set_facecolors = facecolors

# set color of all text to black
for i in ax.findobj(Text):
    i.set_color('black')
#==================================End of graphical customization code==================================  

#Labels
labels = ['Monday', 'Tuesday', 'Wednesday',' Thursday', 'Friday', 'Saturday', 'Sunday', 'Total']
ax.set_xticklabels(labels)
ax.xaxis.tick_top()
plt.yticks(rotation=0)
plt.title('Number of incidents by weekday and hour for years 2012-2017', fontsize=16, y=1.05)
plt.xlabel('Incident weekday', fontsize=16)
plt.ylabel('Incident hour', fontsize=16);
del pivoted_table
# Creating columns for day of month
timematrix = pd.DataFrame()
timematrix['Day'] = pd.DatetimeIndex(crime.index).day
# Manipulating data to feed pivot table to then feed to seaborn heatmap
incidents_md = crime.groupby(['Day','Month'])['Total Incidents'].sum()
incidents_md = incidents_md.reset_index(level=[0,1]) # to go from multiIndex to singleIndex
pivoted_table = incidents_md.pivot(index='Month', columns='Day', values='Total Incidents')
pivoted_table.fillna(0, inplace=True)

#Select max value from the data
max_value = pivoted_table.max().max()

#Create sum for rows and columns
pivoted_table.loc['Total'] = pivoted_table.sum()
pivoted_table = pd.concat([pivoted_table,pd.DataFrame(pivoted_table.sum(axis=1),columns=['Total'])],axis=1)

#Order table by month name
labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Total']
pivoted_table = pivoted_table.reindex(labels)

#Plot
plt.figure(figsize=(18, 8), dpi=90)
ax = sns.heatmap(pivoted_table, cmap='Greens', annot=True, fmt='g', annot_kws={'size': 9}, vmax=max_value)

#==================================Graphical customization code=======================================
#Set white color to total column and row
# find your QuadMesh object and get array of colors
quadmesh = ax.findobj(QuadMesh)[0]
facecolors = quadmesh.get_facecolors()

# make colors of the last column white
column_number = pivoted_table.shape[1]
cells_number = pivoted_table.shape[0]*pivoted_table.shape[1]
last_row = pivoted_table.shape[1]*(pivoted_table.shape[0]-1)

facecolors[np.arange(column_number-1,cells_number,column_number)] = np.array([1,1,1,1]) #change column total to white
facecolors[np.arange(last_row, cells_number,1)] = np.array([1,1,1,1]) #change row total to white

# set modified colors
quadmesh.set_facecolors = facecolors

# set color of all text to black
for i in ax.findobj(Text):
    i.set_color('black')
#==================================End of graphical customization code==================================  

ax.xaxis.tick_top()
plt.yticks(rotation=0)
plt.title('Number of incidents by month and day for years 2012-2017', fontsize=16, y=1.05)
plt.xlabel('Incident day', fontsize=16)
plt.ylabel('Incident month', fontsize=16);

#select locations with at least 300 incidents recorder
t = crime.groupby(['Location 1', 'Longitude', 'Latitude'])[['Location 1']].count()
t = t.sort_values(by=['Location 1'], ascending=False)
t = t[t['Location 1'] >= 300]
t = t.reset_index(level=[1,2])
t