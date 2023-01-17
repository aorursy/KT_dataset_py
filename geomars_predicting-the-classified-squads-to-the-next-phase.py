import os 

import pandas as pd # data processing
#import numpy as np # linear algebra
from scipy import stats #zscore
import warnings
warnings.filterwarnings("ignore")

# Visualization
import matplotlib.pyplot as plt #plot
import seaborn as sns
import plotly.plotly as py #plotly
from plotly.offline import init_notebook_mode, iplot #plotly
init_notebook_mode(connected=True) #plotly
import plotly.graph_objs as go #plotly
worldcup_general = pd.read_csv('../input/fifa-world-cup/WorldCups.csv', index_col='Year') #reading the dataset
# Removing points in the values and converting to integer
worldcup_general['Attendance'] = worldcup_general['Attendance'].str.replace('\.', "").astype('int64') 

winner = worldcup_general['Winner'].value_counts() #Counting the countries who have win the world cup
second_place = worldcup_general['Runners-Up'].value_counts() #Counting the countries who ended a world cup in the second place
third_place = worldcup_general['Third'].value_counts() #Counting the countries who ended a world cup in the third place
fourth_place = worldcup_general['Fourth'].value_counts() #Counting the countries who ended a world cup in the fourth place

#Concatenating the dataframes - Row-Wise
overall_results_df = pd.concat([winner,second_place,third_place,fourth_place], 
                               keys=['Champion','Second','Third','Fourth']).reset_index()

#Renaming the columns
overall_results_df.columns = ['Performance', 'Country','Count']
#Changing 'Germany FR" to "Germany"
overall_results_df['Country'] = overall_results_df['Country'].replace('Germany FR', 'Germany')
#tranforming the dataframe to plot - Pivoting Performance Variable, filling "NaNs" and sorting by the champion column.
overall_results_wide = overall_results_df.pivot_table(index='Country', 
                                                      columns='Performance', 
                                                      values='Count').fillna(0)

overall_results_wide.head(10)
#
overall_results_wide[['Champion','Second','Third','Fourth']].\
        plot(kind='barh',layout=(1,4),subplots=True, sort_columns=True, legend=False,
        figsize=(15,9), fontsize=10, sharex=True, sharey=True, title='Best countries - All World Cups')

plt.style.use('seaborn')
plt.show()
worldcup_general[['QualifiedTeams','MatchesPlayed']].plot(kind='bar', figsize=(11,7), subplots=True, fontsize=13)

plt.style.use('seaborn')
plt.show()
matches = pd.read_csv('../input/fifa-world-cup/WorldCupMatches.csv')

matches = matches.drop_duplicates()
matches['Year'] = matches['Year'].dropna().astype('int64')
worldcup_general[worldcup_general['Country'] == worldcup_general['Winner']] 
color2 = ['#b73550','#003366','#003366','#b73550','#003366','#003366','#b73550','#003366','#b73550','#003366',
          '#003366',  '#003366','#b73550','#003366','#b73550', '#df6703','#003366','#003366','#50fa83','#b73550']

x_data2 = ['1930','1934','1938','1950','1954',
          '1958','1962','1966','1970','1974',
          '1978','1982','1986','1990','1994',
          '1998','2002','2006','2010','2014']
             
y1930 = matches[matches['Year'] == 1930]['Attendance']
y1934 = matches[matches['Year'] == 1934]['Attendance']
y1938 = matches[matches['Year'] == 1938]['Attendance']
y1950 = matches[matches['Year'] == 1950]['Attendance']
y1954 = matches[matches['Year'] == 1954]['Attendance']
y1958 = matches[matches['Year'] == 1958]['Attendance']
y1962 = matches[matches['Year'] == 1962]['Attendance']
y1966 = matches[matches['Year'] == 1966]['Attendance']
y1970 = matches[matches['Year'] == 1970]['Attendance']
y1974 = matches[matches['Year'] == 1974]['Attendance']
y1978 = matches[matches['Year'] == 1978]['Attendance']
y1982 = matches[matches['Year'] == 1982]['Attendance']
y1986 = matches[matches['Year'] == 1986]['Attendance']
y1990 = matches[matches['Year'] == 1990]['Attendance']
y1994 = matches[matches['Year'] == 1994]['Attendance']
y1998 = matches[matches['Year'] == 1998]['Attendance']
y2002 = matches[matches['Year'] == 2002]['Attendance']
y2006 = matches[matches['Year'] == 2006]['Attendance']
y2010 = matches[matches['Year'] == 2010]['Attendance']
y2014 = matches[matches['Year'] == 2014]['Attendance']

y_data2 = [y1930,y1934,y1938,y1950,y1954,y1958,y1962,
        y1966,y1970,y1974,y1978,y1982,y1986,y1990,
        y1994,y1998,y2002,y2006,y2010,y2014]

traces = []

layout = go.Layout(title='Attendance in matches by World Cup',
                   paper_bgcolor='white',
                   plot_bgcolor='white',
                   showlegend=False
)

for name, values, cls in zip(x_data2, y_data2, color2): 
    traces.append(go.Box(
        y=values,
        name=name,
        whiskerwidth=0.75,
        jitter=0.15,
        marker=dict(
            color=cls,
            size=5.5
        )))  

fig = go.Figure(data=traces, layout=layout)

iplot(fig)
worldcup_general['Average Attendance'] = worldcup_general['Attendance'].div(worldcup_general['MatchesPlayed'])

worldcup_general['Continent'] = ['America','Europe','Europe','America','Europe','Europe','America','Europe',
                                'America','Europe','America','Europe','America','Europe','America','Europe',
                                'Asia','Europe','Africa','America']

worldcup_general.groupby('Continent').mean()['Average Attendance']
#Countries population 
worldcup_general['Population'] = [4200000,41651000,42000000,54000000,5000000,7500000,8000000,42000000,52000000,79000000, 27500000, 
                                  38000000, 77400000, 56500000, 263000000, 58000000, 150000000,82000000, 56000000, 203000000]
# Note that I have researched the population of the countries in the year that the world cups was hosted
worldcup_general['Population - Home Country in millions'] = worldcup_general['Population'] / 1000000  #Dividing by one million

#Now I'll create a scatterplot to check if thereis a linear relationship
worldcup_general.plot(kind='scatter', x = 'Population - Home Country in millions', y = 'Average Attendance', figsize=(9,9),
                      alpha = 0.7, fontsize=15)
plt.title('Population in millions vs Average attendance in World Cup', fontsize=18)
plt.xlabel('Population in millions', fontsize=15)
plt.ylabel('Average Attendance', fontsize=15)
plt.style.use('seaborn')
plt.show()
worldcup_general['Population - Home Country in millions'].corr(worldcup_general['Average Attendance'])
market_value = pd.read_csv('../input/complementary-dataset-fifa-world-cup/market_value_.csv', index_col=['player','country'])

market_value['market_value'] = market_value['market_value'].str.split(' ')
market_value['market_value'] = market_value['market_value'].str[0]
market_value['market_value'] = market_value['market_value'].str.replace(',', '.').astype('float64')

market_value.columns = ['group','position','position_in_general','team','market_value (millions of euros)']
#selecting columns
market_value_sub = market_value[['market_value (millions of euros)']]
#reseting index
market_value_sub = market_value_sub.reset_index()
#setting the name of the players as the index
market_value_sub = market_value_sub.set_index('player')
market_value_sub.plot(kind='hist', bins=40, figsize=[8,5], legend=False)
plt.ylabel('number of players',fontsize=13)
plt.xlabel('market value',fontsize=13)
plt.title('Number of players by market value', fontsize=22)
market_value.groupby('position').median()['market_value (millions of euros)'].\
        plot(kind='barh', figsize=[10,9], color='#f3dc8a', edgecolor=['black']*13, fontsize=15)

plt.xlabel('position', fontsize=19)
plt.ylabel('median market value',fontsize=19)
plt.title('Median market value by position',fontsize=22)

market_value_sub['country'] = market_value_sub['country'].astype('category')
market_value_sub['country_abrev'] = market_value_sub['country'].str[0:3].str.upper()

x_data = ['Russia','Uruguay','Saudi Arabia','Egypt',
          'Morocco','Iran','Portugal','Spain',
          'France','Peru','Denmark','Australia',
          'Argentina','Croatia','Nigeria','Iceland',
          'Switzerland','Brazil','Serbia','Costa Rica',
          'Germany','Sweden','Mexico','South Korea',
          'England','Belgium','Panama','Tunisia',
          'Japan','Senegal','Poland','Colombia']       

x0 = market_value_sub[market_value_sub['country'] == 'Russia']['market_value (millions of euros)']
x1 = market_value_sub[market_value_sub['country'] == 'Uruguay']['market_value (millions of euros)']
x2 = market_value_sub[market_value_sub['country'] == 'Saudi Arabia']['market_value (millions of euros)']
x3 = market_value_sub[market_value_sub['country'] == 'Egypt']['market_value (millions of euros)']
x4 = market_value_sub[market_value_sub['country'] == 'Morocco']['market_value (millions of euros)']
x5 = market_value_sub[market_value_sub['country'] == 'Iran']['market_value (millions of euros)']
x6 = market_value_sub[market_value_sub['country'] == 'Portugal']['market_value (millions of euros)']
x7 = market_value_sub[market_value_sub['country'] == 'Spain']['market_value (millions of euros)']
x8 = market_value_sub[market_value_sub['country'] == 'France']['market_value (millions of euros)']
x9 = market_value_sub[market_value_sub['country'] == 'Peru']['market_value (millions of euros)']
x10 = market_value_sub[market_value_sub['country'] == 'Denmark']['market_value (millions of euros)']
x11 = market_value_sub[market_value_sub['country'] == 'Australia']['market_value (millions of euros)']
x12 = market_value_sub[market_value_sub['country'] == 'Argentina']['market_value (millions of euros)']
x13 = market_value_sub[market_value_sub['country'] == 'Croatia']['market_value (millions of euros)']
x14 = market_value_sub[market_value_sub['country'] == 'Nigeria']['market_value (millions of euros)']
x15 = market_value_sub[market_value_sub['country'] == 'Iceland']['market_value (millions of euros)']
x16 = market_value_sub[market_value_sub['country'] == 'Switzerland']['market_value (millions of euros)']
x17 = market_value_sub[market_value_sub['country'] == 'Brazil']['market_value (millions of euros)']
x18 = market_value_sub[market_value_sub['country'] == 'Serbia']['market_value (millions of euros)']
x19 = market_value_sub[market_value_sub['country'] == 'Costa Rica']['market_value (millions of euros)']
x20 = market_value_sub[market_value_sub['country'] == 'Germany']['market_value (millions of euros)']
x21 = market_value_sub[market_value_sub['country'] == 'Sweden']['market_value (millions of euros)']
x22 = market_value_sub[market_value_sub['country'] == 'Mexico']['market_value (millions of euros)']
x23 = market_value_sub[market_value_sub['country'] == 'South Korea']['market_value (millions of euros)']
x24 = market_value_sub[market_value_sub['country'] == 'England']['market_value (millions of euros)']
x25 = market_value_sub[market_value_sub['country'] == 'Belgium']['market_value (millions of euros)']
x26 = market_value_sub[market_value_sub['country'] == 'Panama']['market_value (millions of euros)']
x27 = market_value_sub[market_value_sub['country'] == 'Tunisia']['market_value (millions of euros)']
x28 = market_value_sub[market_value_sub['country'] == 'Japan']['market_value (millions of euros)']
x29 = market_value_sub[market_value_sub['country'] == 'Senegal']['market_value (millions of euros)']
x30 = market_value_sub[market_value_sub['country'] == 'Poland']['market_value (millions of euros)']
x31 = market_value_sub[market_value_sub['country'] == 'Colombia']['market_value (millions of euros)']

y_data = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,
         x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31]

color = ['#8e0e0e','#8e0e0e','#8e0e0e','#8e0e0e','#50fa83','#50fa83','#50fa83','#50fa83',
         '#6300a5','#6300a5','#6300a5','#6300a5','#b5c3cb','#b5c3cb','#b5c3cb','#b5c3cb',
         '#ffc100','#ffc100','#ffc100','#ffc100','#00cadb','#00cadb','#00cadb','#00cadb',
         '#fa8350','#fa8350','#fa8350','#fa8350','#2e8b57','#2e8b57','#2e8b57','#2e8b57']

layout = go.Layout(
    title='Country Squads - Market Value',
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=False
)

traces = []

for xd, yd, cls in zip(x_data, y_data, color):
    traces.append(go.Box(
        y=yd,
        name=xd,
        marker= dict(
            color=cls,
            size=5
        )
    ))
    
fig = go.Figure(data=traces, layout=layout)
iplot(fig)
median_market_value = market_value[['group', 'market_value (millions of euros)']].reset_index()
median_market_value =  median_market_value[['country','group','market_value (millions of euros)']]
median_market_value = median_market_value.groupby(['group','country']).median()['market_value (millions of euros)']
median_market_value = median_market_value.reset_index().set_index('country')
f, axes = plt.subplots(ncols=4, nrows=2, figsize=(11, 11), sharey=True)

median_market_value[median_market_value['group'] == 'A']['market_value (millions of euros)'].plot(
    kind="bar", ax=axes[0,0], fontsize=13, title='Group A', color='#f3dc8a', edgecolor=['black']*4)

median_market_value[median_market_value['group'] == 'B']['market_value (millions of euros)'].plot(kind="bar", ax=axes[0,1], fontsize=13, 
                                                                title='Group B', color='#f3dc8a', edgecolor=['black']*4)

median_market_value[median_market_value['group'] == 'C']['market_value (millions of euros)'].plot(kind="bar", ax=axes[0,2], fontsize=13, 
                                                                title='Group C', color='#f3dc8a', edgecolor=['black']*4)

median_market_value[median_market_value['group'] == 'D']['market_value (millions of euros)'].plot(kind="bar", ax=axes[0,3], fontsize=13, 
                                                                title='Group D', color='#f3dc8a', edgecolor=['black']*4)

median_market_value[median_market_value['group'] == 'E']['market_value (millions of euros)'].plot(kind="bar", ax=axes[1,0], fontsize=13, 
                                                                title='Group E', color='#f3dc8a', edgecolor=['black']*4)

median_market_value[median_market_value['group'] == 'F']['market_value (millions of euros)'].plot(kind="bar", ax=axes[1,1], fontsize=13, 
                                                                title='Group F', color='#f3dc8a', edgecolor=['black']*4)

median_market_value[median_market_value['group'] == 'G']['market_value (millions of euros)'].plot(kind="bar", ax=axes[1,2], fontsize=13, 
                                                                title='Group G', color='#f3dc8a', edgecolor=['black']*4)

median_market_value[median_market_value['group'] == 'H']['market_value (millions of euros)'].plot(kind="bar", ax=axes[1,3], fontsize=13, 
                                                                title='Group H', color='#f3dc8a', edgecolor=['black']*4)
axes[0,0].set_ylabel('Median Market Value')
axes[1,0].set_ylabel('Median Market Value')

plt.tight_layout()

plt.style.use('seaborn')
plt.show()
ranking_countries = pd.read_csv('../input/complementary-dataset-fifa-world-cup/ranking_countries.csv')

ranking_countries = ranking_countries[['country','group_2018','total_ranking']]

ranking_countries = ranking_countries.set_index('country')

ranking_countries.head()

ranking_countries.loc['South Korea', 'total_ranking'] = 135 #The South Korea value was wrong.
ranking_countries['Average Performance in Cups'] = ranking_countries['total_ranking'] // 5

ranking_countries = ranking_countries.reset_index().sort_values(by='country').set_index('country')

ranking_countries[['Average Performance in Cups']].plot(kind='bar', figsize=[18,6], fontsize=15, color = '#f3dc8a', edgecolor=(['black']*32), legend=False)

plt.xlabel('Countries', fontsize=15)
plt.ylabel('Average Performance in Cups', fontsize = 15)
plt.style.use('fast')
plt.title('Average Performance in Cups by Country', fontsize=15)
df = ranking_countries.copy()

f, axes = plt.subplots(ncols=4, nrows=2, figsize=(11, 11), sharey=True)

df[df['group_2018'] == 'A']['Average Performance in Cups'].plot(kind="bar", ax=axes[0,0], fontsize=15, 
                                                            title='Group A', color='#f3dc8a', edgecolor=['black']*4)


df[df['group_2018'] == 'B']['Average Performance in Cups'].plot(kind="bar", ax=axes[0,1], fontsize=15, 
                                                                title='Group B', color='#f3dc8a', edgecolor=['black']*4)

df[df['group_2018'] == 'C']['Average Performance in Cups'].plot(kind="bar", ax=axes[0,2], fontsize=15, 
                                                                title='Group C', color='#f3dc8a', edgecolor=['black']*4)

df[df['group_2018'] == 'D']['Average Performance in Cups'].plot(kind="bar", ax=axes[0,3], fontsize=15, 
                                                                title='Group D', color='#f3dc8a', edgecolor=['black']*4)

df[df['group_2018'] == 'E']['Average Performance in Cups'].plot(kind="bar", ax=axes[1,0], fontsize=15, 
                                                                title='Group E', color='#f3dc8a', edgecolor=['black']*4)

df[df['group_2018'] == 'F']['Average Performance in Cups'].plot(kind="bar", ax=axes[1,1], fontsize=15, 
                                                                title='Group F', color='#f3dc8a', edgecolor=['black']*4)

df[df['group_2018'] == 'G']['Average Performance in Cups'].plot(kind="bar", ax=axes[1,2], fontsize=15, 
                                                                title='Group G', color='#f3dc8a', edgecolor=['black']*4)

df[df['group_2018'] == 'H']['Average Performance in Cups'].plot(kind="bar", ax=axes[1,3], fontsize=15, 
                                                                title='Group H', color='#f3dc8a', edgecolor=['black']*4)
axes[0,0].set_ylabel('Average Performance in World Cups', fontsize=15)
axes[1,0].set_ylabel('Average Performance in World Cups', fontsize=15)

plt.tight_layout()

plt.style.use('seaborn')
plt.show()
df2 = median_market_value.copy() #copying the dataset to reduce the amount of code needed
df = df.reset_index() # reseting index of the df1
df2 = df2.reset_index() # reseting index of the df2


df3 = pd.merge(df, df2, how='inner', left_on=['country','group_2018'],right_on=['country','group']) #Inner Join
df3 = df3[['country','group','total_ranking','Average Performance in Cups','market_value (millions of euros)']] #selecting columns
df3.head() #showing the head of the new df
#Standardizing the columns, before tooking the average (The two columns have different measures)
df3['Average Performance in Cups - Z'] = stats.zscore(df['Average Performance in Cups']) #Standardizing the Average Performance in Cups
df3['median_market_value_Z'] = stats.zscore(df3['market_value (millions of euros)']) #Standardizing the Median Market value by country
df3['combined_scores_Z'] = (df3['median_market_value_Z'] + df3['Average Performance in Cups - Z'])/2 #Combining them and tooking the average
df3.head()
df3 = df3.set_index('country')

f, axes = plt.subplots(ncols=4, nrows=2, figsize=(11, 11), sharey=True) #Setting up the figure

#Creating subplots
df3[df3['group'] == 'A']['combined_scores_Z'].plot(kind="bar", ax=axes[0,0], fontsize=16, 
                                                            title='Group A', color='#f3dc8a', edgecolor=['black']*4)
df3[df3['group'] == 'B']['combined_scores_Z'].plot(kind="bar", ax=axes[0,1], fontsize=16, 
                                                                title='Group B', color='#f3dc8a', edgecolor=['black']*4)

df3[df3['group'] == 'C']['combined_scores_Z'].plot(kind="bar", ax=axes[0,2], fontsize=16, 
                                                                title='Group C', color='#f3dc8a', edgecolor=['black']*4)

df3[df3['group'] == 'D']['combined_scores_Z'].plot(kind="bar", ax=axes[0,3], fontsize=16, 
                                                                title='Group D', color='#f3dc8a', edgecolor=['black']*4)

df3[df3['group'] == 'E']['combined_scores_Z'].plot(kind="bar", ax=axes[1,0], fontsize=16, 
                                                                title='Group E', color='#f3dc8a', edgecolor=['black']*4)

df3[df3['group'] == 'F']['combined_scores_Z'].plot(kind="bar", ax=axes[1,1], fontsize=16, 
                                                                title='Group F', color='#f3dc8a', edgecolor=['black']*4)

df3[df3['group'] == 'G']['combined_scores_Z'].plot(kind="bar", ax=axes[1,2], fontsize=16, 
                                                                title='Group G', color='#f3dc8a', edgecolor=['black']*4)

df3[df3['group'] == 'H']['combined_scores_Z'].plot(kind="bar", ax=axes[1,3], fontsize=16, 
                                                                title='Group H', color='#f3dc8a', edgecolor=['black']*4)
#setting the yaxis label
axes[0,0].set_ylabel('Combined Indicators (Z Score)', fontsize=15)
axes[1,0].set_ylabel('Combined Indicators (Z Score)', fontsize=15)
#Improving the layout
plt.tight_layout()

plt.style.use('seaborn')
plt.show() #showing the plot
observed_results = set(['Uruguay','Russia','Spain','Portugal','France','Denmark','Argentina','Croatia',
                        'Brazil','Switzerland','Mexico','Sweden','England','Belgium','Japan','Colombia'])

expected_results_combination_of_scores = ['Uruguay','Russia','Spain','Portugal','Denmark','France','Argentina','Croatia',
                                          'Brazil','Switzerland','Germany','Mexico','England','Belgium','Japan','Colombia']

expected_results_average_performance_in_cups = ['Uruguay','Saudi Arabia','Spain','Portugal','France','Denmark','Argentina',
                                                'Croatia','Brazil','Costa Rica','Germany','Mexico','England','Belgium','Japan','Colombia']

expected_results_median_market_value = ['Uruguay','Russia','Spain','Portugal','France','Denmark','Argentina','Croatia','Brazil',
                                        'Serbia','Germany','Mexico','England','Belgium','Senegal']

#Comparing the expected results (Combination of Historical performance and median market value) versus the observed results
combined_scores = []
error_combined_scores = []
for c in observed_results:
    for d in expected_results_combination_of_scores:
        if d == c:
            combined_scores.append(d)

#Comparing the expected results (Median Market Value) versus the observed results
market_value_scores = []
not_market_value_scores = []
for c in observed_results:
    for d in expected_results_median_market_value:
        if d == c:
            market_value_scores.append(d)

average_historical_scores = []
not_average_historical_scores = []
for c in observed_results: 
    for d in expected_results_median_market_value: 
        if d == c: # if the cylinder level type matches,
            average_historical_scores.append(d)

x = 'The average historical performance in World Cups predicted {} out of 16 teams \nThe median market value of the squads predicted {} out of 16 teams \nThe combination of those two variables preditcted {} out of 16 teams'.format(len(average_historical_scores), len(market_value_scores), len(combined_scores))
print(x)