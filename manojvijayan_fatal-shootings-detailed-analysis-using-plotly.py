# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import cufflinks as cf
cf.go_offline()

from functools import reduce

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_income = pd.read_csv("..//input/MedianHouseholdIncome2015.csv", encoding="windows-1252")
df_schooling = pd.read_csv("..//input/PercentOver25CompletedHighSchool.csv", encoding="windows-1252")
df_race = pd.read_csv("..//input/ShareRaceByCity.csv", encoding="windows-1252")
df_poverty = pd.read_csv("..//input/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")
df_killing = pd.read_csv("..//input/PoliceKillingsUS.csv", encoding="windows-1252")
df_income.head()
df_income.info()
def heatMap(df):
    df_heat = pd.DataFrame()
    for each in df.columns:
         df_heat[each] = df[each].apply(lambda x: 1 if pd.notnull(x) else 0)
    df_heat.iplot(kind='heatmap')
heatMap(df_income[df_income['Geographic Area'] == 'WY'])
df_income['Median Income'] = df_income['Median Income'].apply(lambda x : pd.to_numeric(x,errors='coerce'))
df_income.fillna(0,axis=1, inplace=True)
df_income['Geographic Area'] = df_income['Geographic Area'].astype('category')
df_income['City'] = df_income['City'].astype('category')
df_income['Area Rank- Income'] = df_income.groupby('Geographic Area')['Median Income'].rank(ascending=False,method='dense')
df_income['National Rank- Income'] = df_income['Median Income'].rank(ascending=False,method='dense')
total = int(df_income['National Rank- Income'].max())
df_income['Text1'] = df_income.apply(lambda x: "<b>City: {}</b><br><b>National Rank Income: {:,.0f} ({})</b>".format(x['City'], x['National Rank- Income'],total), axis=1)
df_schooling.head()
df_schooling.info()
df_schooling['percent_completed_hs'] = df_schooling['percent_completed_hs'].apply(lambda x : pd.to_numeric(x,errors='coerce'))
heatMap(df_schooling)
df_schooling.fillna(0,axis=1, inplace=True)
df_schooling['National Rank- Schooling'] = df_schooling['percent_completed_hs'].rank(ascending=False,method='dense')
df_schooling['Geographic Area'] = df_schooling['Geographic Area'].astype('category')
df_schooling['City'] = df_schooling['City'].astype('category')
#following lookup stores the total number of cities grouped by area
lkup = dict(df_schooling.groupby('Geographic Area').size())
    
#map the lookup to indivigual rows
df_schooling['Total Cities'] = df_schooling['Geographic Area'].map(lkup)

# create catogories
df_schooling['Geographic Area'] = df_schooling['Geographic Area'].astype('category')

# select all the cities with 100% pass rate
df_top_city_schooling = df_schooling[df_schooling['National Rank- Schooling'] == 1]

#select the least rank present, this would be used in the hoverinfo later
total = int(df_schooling['National Rank- Schooling'].max())

#count the number of cities in each area that have 100% pass rate 
lkup2 = dict(df_top_city_schooling.groupby('Geographic Area').size())
    
#map the lookup to indivigual rows    
df_schooling['No. of Top Cities'] = df_schooling['Geographic Area'].map(lkup2)

#create the text fields required for the hoverinfo.
df_schooling['Text2'] = df_schooling.apply(lambda x: "<b>Cities with pass % 100: {:,.0f}</b><br><b>Total Cities: {:,.0f}</b>".format(x['No. of Top Cities'], x['Total Cities']), axis=1)
df_schooling['Text2_1'] = df_schooling.apply(lambda x: "<br><b>National Rank Education: {:,.0f} ({})</b><br>".format(x['National Rank- Schooling'], total), axis=1)

traces = (go.Bar(y=  (df_top_city_schooling.groupby('Geographic Area').size()*100/df_schooling.groupby('Geographic Area').size()).values,
                     x = [each[0] for each in lkup.items()],
                     text = df_schooling.groupby(['Geographic Area','Text2']).size().reset_index()['Text2'].values,
                     hoverinfo = "text",
                     marker = dict(color='rgb(111, 198, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                 opacity=0.6
              ))
layout = dict(title = "Number of Cities that have 100% High School Pass Rate for 25 and above year olds, Area Wise",
             xaxis = dict(dict(title = 'Area')),
             yaxis = dict(dict(title = "(# of Cities with Pass % 100/Total # of Cities in that Area) % ")))
fig = dict(data = [traces], layout=layout)
iplot(fig)
df_race.info()
#convert the numeric fields to numeric and coerce nonumeric to NaN
cols = df_race.columns[2:]
for each in cols:
    df_race[each] = df_race[each].apply(lambda x : pd.to_numeric(x,errors='coerce'))
#fix the column name
df_race.rename(columns={'Geographic area':'Geographic Area'}, inplace=True)
#set categories
df_race['Geographic Area'] = df_race['Geographic Area'].astype('category')
df_race['City'] = df_race['City'].astype('category')
df_race.info()
#create the text field for the hoverinfo
df_race['Text3'] = df_race.apply(lambda x: "<b>White: {}%</b><br><b>Black: {}%</b><br><b>Native: {}%</b><br><b>Asians: {}%</b><br><b>Hispanic: {}%</b><br>".
                                 format(x['share_white'], x['share_black'],
                                       x['share_native_american'], x['share_asian'],
                                       x['share_hispanic']), axis=1)
df_poverty.info()
df_poverty['poverty_rate'] = df_poverty['poverty_rate'].apply(lambda x : pd.to_numeric(x,errors='coerce'))
df_poverty['Geographic Area'] = df_poverty['Geographic Area'].astype('category')
df_poverty['City'] = df_poverty['City'].astype('category')
heatMap(df_poverty)
df_poverty['poverty_rate'].fillna(0,axis=0, inplace=True)
df_poverty.info()
#create Rank fields
df_poverty['Area Rank- Poverty'] = df_poverty.groupby('Geographic Area')['poverty_rate'].rank(ascending=False,method='dense')
df_poverty['National Rank- Poverty'] = df_poverty['poverty_rate'].rank(ascending=False,method='dense')
total = int(df_poverty['National Rank- Poverty'].max())
df_poverty['Text4'] = df_poverty.apply(lambda x: "<b>National Rank Poverty: {:,.0f} ({})</b><br>".format(x['National Rank- Poverty'],total), axis=1)
df_income_schooling_race_poverty = reduce(lambda left,right: pd.merge(left,right,on=['Geographic Area', 'City'], how='left'), [df_income, df_schooling, df_race, df_poverty])
#select the top city with best median income from each region
df_top_city_income_final = df_income_schooling_race_poverty[df_income_schooling_race_poverty['Area Rank- Income'] == 1].set_index(['Geographic Area'])
df_top_city_income_final.fillna(' ', inplace=True)
#create a trace for the top cities with best median income
traces1 = (go.Bar(y=  df_top_city_income_final['Median Income'],
                     x = df_top_city_income_final.index,
                     text = df_top_city_income_final['Text1'] + df_top_city_income_final['Text2_1'] + df_top_city_income_final['Text3'] + df_top_city_income_final['Text4'],
                     hoverinfo = "text",
                     marker = dict(color='rgb(111, 198, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                     name = 'Best',
                     opacity=0.6
                 )
              )
#create a lookup, area wise, with worse performing Cities.
lkup3 = dict(df_income_schooling_race_poverty[df_income_schooling_race_poverty['Median Income'] != 0].groupby('Geographic Area')['Area Rank- Income'].max())
#select the indices that match the criteria
loc = [df_income_schooling_race_poverty[(df_income_schooling_race_poverty['Geographic Area'] == Area) & (df_income_schooling_race_poverty['Area Rank- Income'] == Rank)].index[0] for Area, Rank in lkup3.items()]
#pick the cities with worse median income.
df_worst_city_income_final = df_income_schooling_race_poverty.iloc[loc].set_index(['Geographic Area'])
df_worst_city_income_final.fillna(' ', inplace=True)
#create a trace for cities with worst median income from each area
traces2 = (go.Bar(y=  df_worst_city_income_final['Median Income'],
                     x = df_worst_city_income_final.index,
                     text = df_worst_city_income_final['Text1'] + df_worst_city_income_final['Text2_1'] + df_worst_city_income_final['Text3'] + df_worst_city_income_final['Text4'],
                     hoverinfo = "text",
                     marker = dict(color='rgb(198, 91, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                     name= 'Worst',
                     opacity=0.6
                 )
              )
#finally, plot the bar chart
layout = dict(title = "Median Income - Best and Worst Performing City from each State",
             xaxis = dict(dict(title = 'Area')),
             yaxis = dict(dict(title = "Median Income")))
fig = dict(data = [traces1,traces2], layout=layout)
iplot(fig)
#renmae the columns
df_killing.rename(columns={'state':'Geographic Area', 'city': 'City'}, inplace=True)
df_killing.info()
#convert Area to Category
df_killing['Geographic Area'] = df_killing['Geographic Area'].astype('category')
# count number of cities that are in top 50 when it comes to median income and store them into a lookup
lkup4 = dict(df_income_schooling_race_poverty[df_income_schooling_race_poverty['National Rank- Income'] <= 50].groupby('Geographic Area').size())
    
#map the summary back to original dataset
df_income_schooling_race_poverty['National Rank50- Income'] = df_income_schooling_race_poverty['Geographic Area'].map(lkup4)

#create a text field to store the details for hoverinfo
df_income_schooling_race_poverty['Text5']  = df_income_schooling_race_poverty.apply(lambda x: "<br><b>Cities in top 50-Median Income: {:,.0f}</b><br>".
                                 format(x['National Rank50- Income']),
                                        axis=1)

# count the cities which have 75% of the population below poverty line
lkup5 = dict(df_income_schooling_race_poverty[df_income_schooling_race_poverty['poverty_rate'] >= 75].groupby('Geographic Area').size())
    
#map it back to original datset
df_income_schooling_race_poverty['75% below poverty # cities'] = df_income_schooling_race_poverty['Geographic Area'].map(lkup5)

#text field for hoverinfo
df_income_schooling_race_poverty['Text6']  = df_income_schooling_race_poverty.apply(lambda x: "<b>Cities with 75% or more below Poverty: {:,.0f}</b><br>".
                                 format(x['75% below poverty # cities']),
                                        axis=1)
# select the hoverinfo fields from the original dataset, only select one row for each Area.
df_state_text_lkup = df_income_schooling_race_poverty.drop_duplicates(subset=['Geographic Area','Total Cities','Text2','Text5','Text6'])[['Geographic Area','Total Cities','Text2','Text5','Text6']].dropna().set_index('Geographic Area')
#first sum up the fatal shootings for each state and append the hoverinfo to it
df_state_inc = pd.merge(df_killing.groupby(['Geographic Area']).size().to_frame(), df_state_text_lkup, how='left', left_index=True, right_index=True)
df_state_inc.rename(columns={0:'Fatal Shootings'}, inplace=True)
#create a new text field for hoverinfo, this will show the incidents per city for each state
df_state_inc['Text1'] = df_state_inc.apply(lambda x: "<b>Shooting Per City: {:.2f}</b><br>".format(x['Fatal Shootings']/x['Total Cities']),axis=1)
traces  = (go.Bar(y=  df_state_inc['Fatal Shootings'],
                     x = df_state_inc.index,
                     text = df_state_inc['Text1'] + df_state_inc['Text5'] + df_state_inc['Text6'] + df_state_inc['Text2'] ,
                     hoverinfo = "text",
                     marker = dict(color='rgb(198, 91, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                     name= 'Fatal Shootings',
                     opacity=0.6
                 )
              )
layout = dict(title = "Fatal Shootings, State Wise",
             xaxis = dict(dict(title = 'State')),
             yaxis = dict(dict(title = "Number of Fatal Shoorings")))
fig = dict(data = [traces], layout=layout)
iplot(fig)
df_killing['manner_of_death'].unique()
df_killing.groupby('armed').size().sort_values(ascending=False)[:10].iplot(kind='bar',title='Top 10 Most used Weapons')
df_killing.groupby('age').size().iplot(kind='bar',title='Age of People Killed')
df_killing.groupby('gender').size().iplot(kind='bar',title='Gender')
df_killing.groupby('City').size().sort_values(ascending=False)[:11].iplot(kind='bar',title='Top 10 Cities with Most Number of Fatal Shootings')
# Extract the year and month from the date and separate them as different columns. 
df_killing = pd.concat([pd.DataFrame([each for each in df_killing['date'].str.split('/').values.tolist()],
                             columns=['Day', 'Month', 'Year']),df_killing],axis=1)
df_killing['Year'] = df_killing['Year'].apply(lambda x: int(x) + 2000)

# Convert the Date column to datetime
df_killing.date = df_killing.date.apply(lambda x : pd.to_datetime(x,dayfirst=True))

#create day of the week column
df_killing['day_of_week'] = df_killing.date.apply(lambda x: pd.to_datetime(x).weekday())
df_killing.groupby([ 'Year', 'Month'])['date'].size().iplot(title = 'Fatal Shootings Monthly', yTitle = "Number", xTitle = "(Year, Month)" )
df_killing[ 'Year'].value_counts().plot(kind='bar',title = 'Fatal Shootings Yearly',figsize=(8,5))
pd.crosstab(df_killing['Month'], df_killing['Year']).iplot(kind='box', title = 'Fatal Killings', xTitle ='Year',
                                                                    yTitle = 'Number')
df_time = df_killing[['Year', 'Month', 'Day', 'day_of_week', 'date']]
df_time['Count'] = 1
df_time = df_time.groupby(['date','Year', 'Month', 'Day', 'day_of_week']).sum().reset_index()
df_time.set_index('date', inplace=True)
df_time.info()
df_time = df_time.reindex(pd.date_range(start="2015", end="2017-07-31", freq='D'))
df_time.info()
df_time.fillna(0, inplace=True)
df_time = df_time.reset_index()
df_time['Year'] = df_time['index'].apply(lambda x: x.strftime('%Y'))
df_time['Month'] = df_time['index'].apply(lambda x: x.strftime('%m'))
df_time['Day'] = df_time['index'].apply(lambda x: x.strftime('%d'))
df_time['day_of_week'] = df_time['index'].apply(lambda x: x.strftime('%w'))
ax1 = pd.crosstab([df_time['Year'],df_time['Month']], df_time['Day'], values=df_time['Count'], aggfunc='sum').reset_index().drop(['Year','Month'],axis=1).plot(kind='box',figsize=(15,5))
ax1.set_ylabel("Count")
ax1.set_xlabel("Day of the Month")
df_killing.groupby([ 'Day'])['date'].size().iplot(title = 'Fatal Shootings by Day of the Month', yTitle = "Count", xTitle = "Day" )
df = df_time.drop(['index','Year', 'Month', 'Day'], axis=1).reset_index()
ax1 = pd.crosstab(df.index,df.day_of_week,values=df.Count, aggfunc='sum').plot(kind='box',figsize=(10,5))
ax1.set_ylabel("Count")
ax1.set_xlabel("Day of the Week")
ax1.set_xticklabels(['Sun', 'Mon', 'Tue','Wed', 'Thu', 'Fri', 'Sat'])
ax1=pd.crosstab(df_time.Year,df_time.Month,values=df_time.Count, aggfunc='sum').plot(kind='box',figsize=(15,5))
ax1.set_ylabel("Count")
ax1.set_xlabel("Month")
df_killing.groupby([ 'Month'])['date'].size().iplot(title = 'Fatal Shootings by Month', yTitle = "Count", xTitle = "Month" )