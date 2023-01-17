import pandas as pd
olympics = pd.read_csv(r'../input/athlete_events.csv', index_col='ID')
regions = pd.read_csv(r'../input/noc_regions.csv')
olympics = pd.merge(olympics, regions, how='left', on=['NOC'])
olympics.columns
olympics.shape
olympics.describe()
olympics.isnull().sum()
olympics.dtypes
for x in olympics.columns:
    print(olympics[x].value_counts())
    print('='*70)
olympics.head()
import seaborn as sns
import matplotlib.pyplot as plt
f, axes = plt.subplots(1, 3)
f.set_size_inches(15, 5)
sns.boxplot(x='Sex', y='Age', data=olympics, ax=axes[0])
sns.boxplot(x='Sex', y='Height', data=olympics, ax=axes[1])
sns.boxplot(x='Sex', y='Weight', data=olympics, ax=axes[2])
plt.show()
ax = sns.countplot(x='Sex', data=olympics)
plt.title('Distribution of Olympics by sex')
plt.ylabel('Number of olympians')
plt.show()
plt.figure(figsize=(18,10), dpi=90)
ax = sns.countplot(x='Age', data=olympics)
plt.xticks(rotation=45)
plt.xlabel('Age')
plt.ylabel('Number of olympians')
plt.show()
max_age = olympics['Age'].max()
min_age = olympics['Age'].min()
olympics.loc[olympics['Age'].isin([min_age, max_age])]
plt.figure(figsize=(20,10), dpi=90)
ax = sns.countplot(x='Height', data=olympics)
plt.xticks(rotation=45)
plt.xlabel('Height')
plt.ylabel('Number of olympians')
plt.show()
min_height = olympics['Height'].min()
max_height = olympics['Height'].max()
olympics.loc[olympics['Height'].isin([min_height,max_height])]
plt.figure(figsize=(14,7))
ax = sns.scatterplot(y='Height', x='Weight', data=olympics, alpha=0.3)
plt.show()
plt.figure(figsize=(14,7))
ax = sns.jointplot(y='Height', x='Weight', data=olympics, kind='hex')
plt.show()
olympics[['Weight', 'Height']].corr()
#Sport in Summer Olympics:
Sport_Count=olympics.groupby('Sport').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Sport_Count.columns=['Sport','Years']
Sport_Count['Count']=[len(c) for c in Sport_Count['Years']]

Sport_Count = Sport_Count.sort_values(by='Count', ascending=False)
Sport_Count.head(10)
Sport_Count.tail(10)
sports = olympics[olympics['Sport'].isin(Sport_Count.Sport[:10])]
plt.figure(figsize=(15,12), dpi=90)

plt.subplot(311)
plt.title('Boxplots of age, height and weight for top ten sports at the Olympics')
ax=sns.violinplot(x='Sport',y='Age',data=sports)

plt.subplot(312)
ax=sns.violinplot(x='Sport',y='Height',data=sports)

plt.subplot(313)
ax=sns.violinplot(x='Sport',y='Weight',data=sports)

plt.subplots_adjust(wspace = 1, hspace = 0.3,top = 1)

plt.show()
plt.figure(figsize=(15,12), dpi=90)

sns.set_style('darkgrid')

plt.subplot(311)
plt.title('Boxplots of age, height and weight for top ten sports at the Olympics')
ax=sns.boxplot(x='Sport',y='Age',data=sports)

plt.subplot(312)
ax=sns.boxplot(x='Sport',y='Height',data=sports)

plt.subplot(313)
ax=sns.boxplot(x='Sport',y='Weight',data=sports)

plt.subplots_adjust(wspace = 1, hspace = 0.3,top = 1)
#olympics.loc[olympics['Medal'].isin(['Gold', 'Silver', 'Bronze']), 'Number'] = 1
medals_by_year = olympics.groupby(['Year', 'Medal'])['Medal'].count()
medals_by_year = medals_by_year.reset_index(level=[1], name='Count')
medals_by_year.head()
plt.figure(figsize=(15,6))
sns.set_style('darkgrid')
ax = sns.scatterplot(x=medals_by_year.index, y='Count', hue='Medal', data=medals_by_year, legend=False)
ax = sns.lineplot(x=medals_by_year.index, y='Count', hue='Medal', data=medals_by_year)
plt.show()
#New dataframe with grouped regions
NOC_medals = olympics.groupby(['region', 'Year'])['Medal'].count()
NOC_medals = NOC_medals.reset_index(level=[0,1])

#Pivoted table used for heatmap
pivoted_table = NOC_medals.pivot(index='region', columns='Year', values='Medal')
pivoted_table.fillna(0, inplace=True)

#Select max value from the data for better color adjustment in heatmap
max_value = pivoted_table.max().max()

#Create sum for rows and columns
pivoted_table.loc['Total'] = pivoted_table.sum()
pivoted_table = pd.concat([pivoted_table,pd.DataFrame(pivoted_table.sum(axis=1),columns=['Total'])],axis=1)

#Reduce dimension of dataframe by selecting countries with at least 1 won medal
pivoted_table = pivoted_table[pivoted_table['Total'] > 0]
from matplotlib.collections import QuadMesh
from matplotlib.text import Text
import numpy as np

plt.figure(figsize=(30,30), dpi=90)
ax = sns.heatmap(data=pivoted_table, cmap='Reds', annot=True, fmt='g', annot_kws={'size': 8}, vmax=max_value)

#==================================Graphical customization code=======================================
#Set white color to total column and row
#find your QuadMesh object and get array of colors
quadmesh = ax.findobj(QuadMesh)[0]
facecolors = quadmesh.get_facecolors()

#make colors of the last column white
column_number = pivoted_table.shape[1]
cells_number = pivoted_table.shape[0]*pivoted_table.shape[1]
last_row = pivoted_table.shape[1]*(pivoted_table.shape[0]-1)

facecolors[np.arange(column_number-1,cells_number,column_number)] = np.array([1,1,1,1]) #change column total to white
facecolors[np.arange(last_row, cells_number,1)] = np.array([1,1,1,1]) #change row total to white

#set modified colors
quadmesh.set_facecolors = facecolors

#set color of all text to black
for i in ax.findobj(Text):
    i.set_color('black')
#==================================End of graphical customization code==================================

#Label customization
ax.xaxis.tick_top()
plt.title('Medals won by country over the years', fontsize=16, y=1.02)
plt.ylabel('Country', fontsize=16);
#New dataframe with grouped regions
NOC_medals = olympics.groupby(['Sport', 'Year'])['Medal'].count()
NOC_medals = NOC_medals.reset_index(level=[0,1])

#Pivoted table used for heatmap
pivoted_table = NOC_medals.pivot(index='Sport', columns='Year', values='Medal')
pivoted_table.fillna(0, inplace=True)

#Select max value from the data for better color adjustment in heatmap
max_value = pivoted_table.max().max()

#Create sum for rows and columns
pivoted_table.loc['Total'] = pivoted_table.sum()
pivoted_table = pd.concat([pivoted_table,pd.DataFrame(pivoted_table.sum(axis=1),columns=['Total'])],axis=1)

#Reduce dimension of dataframe by selecting countries with at least 1 won medal
pivoted_table = pivoted_table[pivoted_table['Total'] > 0]
from matplotlib.collections import QuadMesh
from matplotlib.text import Text
import numpy as np

plt.figure(figsize=(30,30), dpi=90)
ax = sns.heatmap(data=pivoted_table, cmap='Reds', annot=True, fmt='g', annot_kws={'size': 8}, vmax=max_value)

#==================================Graphical customization code=======================================
#Set white color to total column and row
#find your QuadMesh object and get array of colors
quadmesh = ax.findobj(QuadMesh)[0]
facecolors = quadmesh.get_facecolors()

#make colors of the last column white
column_number = pivoted_table.shape[1]
cells_number = pivoted_table.shape[0]*pivoted_table.shape[1]
last_row = pivoted_table.shape[1]*(pivoted_table.shape[0]-1)

facecolors[np.arange(column_number-1,cells_number,column_number)] = np.array([1,1,1,1]) #change column total to white
facecolors[np.arange(last_row, cells_number,1)] = np.array([1,1,1,1]) #change row total to white

#set modified colors
quadmesh.set_facecolors = facecolors

#set color of all text to black
for i in ax.findobj(Text):
    i.set_color('black')
#==================================End of graphical customization code==================================

#Label customization
ax.xaxis.tick_top()
plt.title('Medals for disciplines over the years', fontsize=16, y=1.02)
plt.ylabel('Discipline', fontsize=16);
olympics_PL = olympics[olympics['region'] == 'Poland']
olympics_PL.columns
plt.figure(figsize=(5,3), dpi=90)

sns.countplot(x='Medal', data=olympics_PL, hue='Sex')
plt.ylabel('Number of medals')
plt.show()
olympics_PL_gt0 = olympics_PL[olympics_PL['Medal'].notnull()]
#This chart will be improved in kernel v2
plt.figure(figsize=(15,7), dpi=90)
palette = sns.color_palette("Dark2_r", 25)
ax = sns.countplot(x='Medal', data=olympics_PL_gt0, hue='Sport', palette=palette)
ax.legend(frameon=False, ncol=5, bbox_to_anchor=(0.9, -0.1))
plt.ylabel('Number of medals')
plt.show()