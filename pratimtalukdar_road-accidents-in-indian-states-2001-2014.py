import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import display as display
import plotly.graph_objs as go
# Read the data to the dataframe from the data files.

#state_year_month_df contains data for each state, segragated into year and month
state_year_month_df=pd.read_csv('../input/only_road_accidents_data_month2.csv')

#state_year_time_df contains data for each state, segragated into year and time of the day
state_year_time_df=pd.read_csv('../input/only_road_accidents_data3.csv')


state_year_month_df.head()
state_year_time_df.head()
#Get all the state names in an array..
state_names=state_year_month_df['STATE/UT'].unique()
print(state_names)
#state_year_month_df=state_year_month_df['STATE/UT']
state_year_month_df['STATE/UT']=state_year_month_df['STATE/UT'].replace({'Delhi (Ut)': 'Delhi Ut', 'D & N Haveli':'D&N Haveli'})
print(state_year_month_df['STATE/UT'].unique())
# Reassiging state names to variable..
state_names=state_year_month_df['STATE/UT'].unique()

#display(state_year_month_df.head())

#Create season groups clubbing values from multiple month columns..
state_year_month_df['SUMMER']=state_year_month_df[['JUNE','JULY','AUGUST']].sum(axis=1)
state_year_month_df['AUTUMN']=state_year_month_df[['SEPTEMBER','OCTOBER','NOVEMBER']].sum(axis=1)
state_year_month_df['WINTER']=state_year_month_df[['DECEMBER','JANUARY','FEBRUARY']].sum(axis=1)
state_year_month_df['SPRING']=state_year_month_df[['MARCH','APRIL','MAY']].sum(axis=1)

#Delete month columns..
state_year_month_df=state_year_month_df.drop(['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY'
                                             ,'AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER'], axis=1)
#Create groups of states, summing the values of accident number for each year..
state_grouped=state_year_month_df.groupby(['STATE/UT']).sum()

#Create % columns for noting the % of accidents happening in each state for each season..
state_grouped['%_SUMMER']=state_grouped['SUMMER']/state_grouped['TOTAL']
state_grouped['%_AUTUMN']=state_grouped['AUTUMN']/state_grouped['TOTAL']
state_grouped['%_WINTER']=state_grouped['WINTER']/state_grouped['TOTAL']
state_grouped['%_SPRING']=state_grouped['SPRING']/state_grouped['TOTAL']

display(state_grouped.iloc[:,1:].head())
#Working on the over the day data...
state_year_time_df.rename(columns={'0-3 hrs. (Night)':'0-3',
                              '3-6 hrs. (Night)':'3-6',
                                '6-9 hrs (Day)':'6-9', '9-12 hrs (Day)':'9-12','12-15 hrs (Day)':'12-15','15-18 hrs (Day)':'15-18',
                                  '18-21 hrs (Night)':'18-21','21-24 hrs (Night)':'21-24'}, inplace=True)
state_time_grouped=state_year_time_df.groupby(['STATE/UT']).sum()

state_time_grouped['%_MORNING']=(state_time_grouped['6-9']+state_time_grouped['9-12'])/state_time_grouped['Total']
state_time_grouped['%_AFTERNOON']=(state_time_grouped['12-15']+state_time_grouped['15-18'])/state_time_grouped['Total']
state_time_grouped['%_EVENING']=(state_time_grouped['18-21']+state_time_grouped['21-24'])/state_time_grouped['Total']
state_time_grouped['%_NIGHT']=(state_time_grouped['0-3']+state_time_grouped['3-6'])/state_time_grouped['Total']

state_time_grouped=state_time_grouped.drop(state_time_grouped.columns[0:9], axis=1)
display(state_time_grouped.head())

plt.figure(figsize=(15,5))
ax=plt.subplot(1,2,1)
boxplot=state_grouped.boxplot(ax=ax,column=['%_SUMMER','%_WINTER','%_AUTUMN','%_SPRING'])

ax=plt.subplot(1,2,2)
state_grouped.loc[:,'SUMMER':'SPRING'].sum(axis=0).plot.pie(title='Seasonal distribution of all accidents in India(2001-14)',autopct='%1.1f%%')

plt.figure(figsize=(20,5))
plt.subplot(141)
summer_sorted=state_grouped.sort_values('%_SUMMER')
summer_sorted['%_SUMMER'].tail(5).plot.bar(title='Highest Summer Accidents')
plt.subplot(142)
winter_sorted=state_grouped.sort_values('%_WINTER')
winter_sorted['%_WINTER'].tail(5).plot.bar(title='Highest Winter Accidents')
plt.subplot(143)
autumn_sorted=state_grouped.sort_values('%_AUTUMN')
autumn_sorted['%_AUTUMN'].tail(5).plot.bar(title='Highest Autumn Accidents')
plt.subplot(144)
spring_sorted=state_grouped.sort_values('%_SPRING')
spring_sorted['%_SPRING'].tail(5).plot.bar(title='Highest Spring Accidents')

highest_accident_states=state_grouped.sort_values('TOTAL', ascending=False)
high_states=list(highest_accident_states.head().index)
df4=state_year_month_df.loc[state_year_month_df['STATE/UT'].isin(high_states),['STATE/UT','YEAR','TOTAL']]

plt.figure(figsize=(10,5))
ax=plt.subplot(111)
for key, grp in df4.groupby(['STATE/UT']):
    ax = grp.plot(ax=ax, kind='line', x='YEAR', y='TOTAL', label=key)
  
plt.show()

highest_accident_states=state_grouped.sort_values('TOTAL', ascending=False)
state_list=list(highest_accident_states.head().index)
print(state_list)

df=state_time_grouped.loc[state_time_grouped.index.isin(state_list)]

df_T=df.groupby('STATE/UT').sum().drop(['Total'], axis=1).T.plot.pie(subplots=True, figsize=(20, 5),autopct='%1.1f%%')
## Break up accidents for all states over the time blocks:
#state_time_grouped.info()
df2=state_time_grouped.sum(axis=0)



df2.drop(['Total']).T.plot.pie(title='All accidents 2001-2014',subplots=True, figsize=(5,5),autopct='%1.1f%%')

df2=state_time_grouped.sum(axis=0)
df3=state_year_time_df.groupby(['YEAR']).sum()
df3.loc[:,'Total'].plot(title='Accidents growth in India')
#See the states with highest % accident in the every timeblock..
plt.figure(figsize=(10,5))
state_time_grouped.sort_values('%_MORNING',ascending=False).head().loc[:,['STATE/UT','%_MORNING']].plot(kind='bar', ax=plt.subplot(221), color='b')
state_time_grouped.sort_values('%_AFTERNOON',ascending=False).head().loc[:,['STATE/UT','%_AFTERNOON']].plot(kind='bar', ax=plt.subplot(222),color='g')
state_time_grouped.sort_values('%_EVENING',ascending=False).head().loc[:,['STATE/UT','%_EVENING']].plot(kind='bar', ax=plt.subplot(223),color='r')
state_time_grouped.sort_values('%_NIGHT',ascending=False).head().loc[:,['STATE/UT','%_NIGHT']].plot(kind='bar', ax=plt.subplot(224),color='y')

#Create a new dataframe - period_performance.
period_performance=pd.DataFrame(columns=['STATE/UT','%_CHANGE_2001_TO_2014'])

#Take one state name at a time,
for state in state_names:
    #print(state)
    total_2001=state_year_month_df.loc[(state_year_month_df['STATE/UT']==state) & (state_year_month_df['YEAR']==2001), 'TOTAL']
    total_2014=state_year_month_df.loc[(state_year_month_df['STATE/UT']==state) & (state_year_month_df['YEAR']==2014), 'TOTAL']
    value_2001=total_2001.iloc[0]
    value_2014=total_2014.iloc[0]
    change_in_percent= (value_2014-value_2001)*100/value_2001
   
    new_data=pd.Series({'STATE/UT':state, '%_CHANGE_2001_TO_2014':change_in_percent})
    period_performance=period_performance.append(new_data, ignore_index=True)
best_performing=period_performance.sort_values('%_CHANGE_2001_TO_2014')
#print(best_performing.head())
ax=best_performing.plot(kind='bar').set_xticklabels(best_performing['STATE/UT'])
