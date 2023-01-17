# Import all of the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
%matplotlib inline
events = pd.read_csv("../input/athlete_events.csv")
events.head(10)
events.info()
events.isnull().sum()
#How many different types of sports do Olympic Games have?
events['Sport'].nunique()
events['Medal'].value_counts(dropna=False)
events.groupby(['Year', 'Medal'])['Medal'].count().unstack().plot(kind='bar',figsize=(15,6))
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
events.groupby(['Year', 'Sex'])[['Height', 'Weight']].mean().unstack().plot(ax=ax1)
ax1.set_title('Average Height and Weight of Sportsmen/women')
events.groupby(['Year', 'Sex'])['Age'].mean().unstack().plot(ax=ax2,figsize=(15,10))
ax1.set_title('Average Age of Sportsmen/women')
#Here is the exact numbers of age difference in the last five OG
events.groupby(['Year', 'Sex'])['Age'].mean().unstack().tail()
grouped_df = events.groupby('Year')[['Height', 'Weight', 'Age']].count()
grouped_df.head(3)
attendees = events['Year'].value_counts().sort_index()
print('The Total Attendees from 1896 to 2016 in the List/Array:')
attendees.values
grouped_df['Total Attendees'] = attendees.values
grouped_df.head()
grouped_df[['Height', 'Weight', 'Total Attendees']].plot(figsize=(15,5), 
          title = 'Number of Existing Height and Weight vs Total Number of Olympic participants')
grouped_df[['Age', 'Total Attendees']].plot(figsize=(15,5))
grouped_df['Height non-NA'] = grouped_df['Height'] / grouped_df['Total Attendees']
grouped_df['Weight non-NA'] = grouped_df['Weight'] / grouped_df['Total Attendees']
grouped_df['Age non-NA'] = grouped_df['Age'] / grouped_df['Total Attendees']
grouped_df.head(3)
grouped_df[['Weight non-NA', 'Height non-NA', 'Age non-NA']].plot(figsize=(15,8), marker='o', alpha = .3, 
                                                 xticks = range(1896, 2018, 6), 
                                                 title='Proportion/Percentage of Non-missing Values for Weight, Height & Age')
events_60up = events.loc[events['Year']>=1960, :]
events_60up.head(3)
events_60up.info()
height_grouped = events_60up.groupby(['Sport', 'Sex'])['Height'].mean().unstack()
height_grouped.sort_values(by='M', ascending=False).head()
height_grouped.sort_values(by='M').plot(kind='barh', figsize=(15,12))
#Practice how to grab a key for getting a value from python dictionary
#Assign height of sportswomen to a variable
f_height = height_grouped.to_dict()['F']
#what is the average value of female basketball player?
f_height['Basketball']
#set a condition which returns all NaN values for rows of female basketball and column of Height
events_60up.loc[(events_60up['Height'].isnull()) & (events_60up['Sex'] == 'F') & (events_60up['Sport'] == 'Basketball'), 'Height'] 
#How to iterate through to get pair of keys and values from python dictionary 
for k,v in f_height.items():
    print(k)
#Plug an average hight of sportswomen for each sport (type).
def f_height_fixer(df):
    for k,v in f_height.items():
        df.loc[(df['Height'].isnull()) & (df['Sex'] == 'F') & (df['Sport'] == str(k)), 'Height'] = v
    return df
#Do we have reasonable numbers of missing values for the height column? 
f_height_fixer(events_60up).info()        
# pandas' .info method tells us that we have more non-null values 
# but lets make sure that it has worked by taking a step further
events_60up.loc[(events_60up['Sex'] == 'F'), 'Height'].isnull().sum()
events_60up.loc[(events_60up['Sex'] == 'M'), 'Height'].isnull().sum()
m_height = height_grouped.to_dict()['M']
def m_height_fixer(df):
    for k,v in m_height.items():
        df.loc[(df['Height'].isnull()) & (df['Sex'] == 'M') & (df['Sport'] == str(k)), 'Height'] = v
    return df
m_height_fixer(events_60up).info() 
events_60up['Height'].unique()
events_60up['Height'] = (round(events_60up['Height'])).astype(int)
events_60up['Height'].unique()
#Let's see the distribution of heights for males and females
events_60up['Height'].hist(by=events_60up['Sex'],figsize=(12,6),sharey=True, bins=25)