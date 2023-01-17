import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
df.head()
df.drop(columns=['fips'], inplace=True)
newest_date = df['date'].max()
df1 = df[df['date'] == newest_date].groupby('date').agg(sum)
df1.reset_index(inplace=True)
df1
df2 = df.groupby('date').agg('sum')
df2.reset_index(inplace=True)
df2[['date', 'cases']]
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.plot(df2['date'], df2['cases'], 'o-', color='steelblue')
#annotate max value
ymax = max(df2['cases'])
xmax = df2.iloc[df2['cases'].idxmax]['date']
plt.annotate(str(ymax), xy=(xmax, ymax), xytext=(0,7), textcoords='offset points')
#for a,b in zip(df4['date'], df4['cases']): 
#    plt.annotate(str(b), xy=(a, b), xytext=(0,5), textcoords='offset points')

ax.set_title('Cases over time', fontsize=20)
ax.set_ylabel('Cases', fontsize=15)
ax.set_xlabel('Date', fontsize=15)
#draw vertical line for every new months
feb = df2[df2['date']=='2020-02-01'].index.values.astype(int)[0]
mar = df2[df2['date']=='2020-03-01'].index.values.astype(int)[0]
apr = df2[df2['date']=='2020-04-01'].index.values.astype(int)[0]
plt.axvline(x=feb, color='indianred')
plt.axvline(x=mar, color='indianred')
plt.axvline(x=apr, color='indianred')
ax.get_xticklabels()[feb].set_color("indianred")
ax.get_xticklabels()[mar].set_color("indianred")
ax.get_xticklabels()[apr].set_color("indianred")
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid(linewidth=0.5)
df2[['date', 'deaths']]
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.plot(df2['date'], df2['deaths'], 'o-', color='steelblue')
#annotate max value
ymax = max(df2['deaths'])
xmax = df2.iloc[df2['deaths'].idxmax]['date']
plt.annotate(str(ymax), xy=(xmax, ymax), xytext=(0,7), textcoords='offset points')
#for a,b in zip(df4['date'], df4['cases']): 
#    plt.annotate(str(b), xy=(a, b), xytext=(0,5), textcoords='offset points')

ax.set_title('Deaths over time', fontsize=20)
ax.set_ylabel('Deaths', fontsize=15)
ax.set_xlabel('Date', fontsize=15)
#draw vertical line for every new months
feb = df2[df2['date']=='2020-02-01'].index.values.astype(int)[0]
mar = df2[df2['date']=='2020-03-01'].index.values.astype(int)[0]
apr = df2[df2['date']=='2020-04-01'].index.values.astype(int)[0]
plt.axvline(x=feb, color='indianred')
plt.axvline(x=mar, color='indianred')
plt.axvline(x=apr, color='indianred')
ax.get_xticklabels()[feb].set_color("indianred")
ax.get_xticklabels()[mar].set_color("indianred")
ax.get_xticklabels()[apr].set_color("indianred")
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid(linewidth=0.5)
df3 = df[df['date'] == newest_date]
df4 = df3.groupby('state').agg('sum').sort_values('cases', ascending=False)
df4.reset_index(inplace=True)
df4[['state', 'cases']].head(10)
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.bar(df4['state'], df4['cases'], color=(df4['cases'] < 25000).map({True: 'steelblue',
                                                                          False: 'indianred'}))
ax.set_title('Cases by state', fontsize=20)
ax.set_ylabel('Total Cases', fontsize=15)
ax.set_xlabel('State', fontsize=15)
y_max_cases = df4['cases'].max()+1000
plt.yticks(np.arange(0, y_max_cases, step=25000))
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid()
df4[['state', 'deaths']].head(10)
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.bar(df4['state'], df4['deaths'], color=(df4['deaths'] < 1000).map({True: 'steelblue',
                                                                          False: 'indianred'}))
ax.set_title('Deaths by state', fontsize=20)
ax.set_ylabel('Total Deaths', fontsize=15)
ax.set_xlabel('State', fontsize=15)
y_max_deaths = df4['deaths'].max()+1000
plt.yticks(np.arange(0, y_max_deaths, step=1000))
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid()
df5 = df.groupby('date').agg('sum')
df5['cases_daily'] = df5['cases'] - df5['cases'].shift(1)
df5['deaths_daily'] = df5['deaths'] - df5['deaths'].shift(1)
df5['cases_daily'].fillna('0', inplace=True)
df5['deaths_daily'].fillna('0', inplace=True)
df5['cases_daily'] = df5['cases_daily'].astype('int64')
df5['deaths_daily'] = df5['deaths_daily'].astype('int64')
df5.reset_index(inplace=True)
df5[['date', 'cases_daily']]
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.plot(df5['date'], df5['cases_daily'], 'o-', color='steelblue')
#annotate max value
ymax = max(df5['cases_daily'])
xmax = df5.iloc[df5['cases_daily'].idxmax]['date']
plt.annotate(str(ymax), xy=(xmax, ymax), xytext=(-3,7), textcoords='offset points', color="r")
#for a,b in zip(df3['date'], df3['cases_daily']): 
#    plt.annotate(str(b), xy=(a, b), xytext=(-3,5), textcoords='offset points')

ax.set_title('Daily new infections since the first occurance', fontsize=20)
ax.set_ylabel('Daily new infections', fontsize=15)
ax.set_xlabel('Date', fontsize=15)
#draw vertical line for every new months
feb = df5[df5['date']=='2020-02-01'].index.values.astype(int)[0]
mar = df5[df5['date']=='2020-03-01'].index.values.astype(int)[0]
apr = df5[df5['date']=='2020-04-01'].index.values.astype(int)[0]
plt.axvline(x=feb, color='indianred')
plt.axvline(x=mar, color='indianred')
plt.axvline(x=apr, color='indianred')
ax.get_xticklabels()[feb].set_color("indianred")
ax.get_xticklabels()[mar].set_color("indianred")
ax.get_xticklabels()[apr].set_color("indianred")
#color xtick red
max_day_y = df5.loc[df5['cases_daily'].idxmax()]['date']
max_day_x = df5[df5['date']==max_day_y].index.values.astype(int)[0]
ax.get_xticklabels()[max_day_x].set_color("red")
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid(linewidth=0.5)
most_infectious_day = df5.groupby('date').agg('sum').sort_values('cases_daily', ascending=False)
most_infectious_day.reset_index(inplace=True)
most_infectious_day[['date', 'cases_daily']].head(1)
#first occurance of death
df5[df5['deaths']>0][['date', 'deaths']].head(2)
date1 = df5.iloc[38:]
date1.reset_index(inplace=True)
date1[['date', 'deaths_daily']].tail(10)
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.plot(date1['date'], date1['deaths_daily'], 'o-', color='steelblue')
#annotate max value
ymax_deaths = max(df5['deaths_daily'])
xmax_deaths = df5.iloc[df5['deaths_daily'].idxmax]['date']
plt.annotate(str(ymax_deaths), xy=(xmax_deaths, ymax_deaths), xytext=(-3,7), textcoords='offset points', color="r")
#for a,b in zip(date1['date'], date1['deaths_daily']): 
#    plt.annotate(str(b), xy=(a, b), xytext=(0,5), textcoords='offset points')

ax.set_title('Daily new Deaths since the first occurance', fontsize=20)
ax.set_ylabel('Daily new Deaths', fontsize=15)
ax.set_xlabel('Date', fontsize=15)
#draw vertical line for every new months
mar = date1[date1['date']=='2020-03-01'].index.values.astype(int)[0]
apr = date1[date1['date']=='2020-04-01'].index.values.astype(int)[0]
plt.axvline(x=mar, color='indianred')
plt.axvline(x=apr, color='indianred')
ax.get_xticklabels()[mar].set_color("indianred")
ax.get_xticklabels()[apr].set_color("indianred")
#color xtick red
max_day_y_death = date1.loc[date1['deaths_daily'].idxmax()]['date']
max_day_x_death = date1[date1['date']==max_day_y_death].index.values.astype(int)[0]
ax.get_xticklabels()[max_day_x_death].set_color("red")
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid(linewidth=0.5)
deadliest_day = df5.groupby('date').agg('sum').sort_values('deaths_daily', ascending=False)
deadliest_day.reset_index(inplace=True)
deadliest_day[['date', 'deaths_daily']].head(1)
mortality_rate = df1['deaths']/df1['cases']*100
mortality_rate
df4['mortality rate'] = round((df4['deaths']/df4['cases'])*100, 2)
df4[['state', 'mortality rate']].head(10)
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.bar(df4['state'], df4['mortality rate'], color=(df4['mortality rate'] < 5).map({True: 'steelblue',
                                                                          False: 'indianred'}))
ax.set_title('Mortality rate by state', fontsize=20)
ax.set_ylabel('Mortality rate', fontsize=15)
ax.set_xlabel('State', fontsize=15)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid()
df6 = df.groupby('date').agg(sum)
df6['mortality rate'] = df6['deaths']/df6['cases']*100
df6.reset_index(inplace=True)
df6[['date', 'mortality rate']]
fig, ax = plt.subplots(figsize=(22,10))
ax.tick_params(axis='x', rotation=70)
ax.plot(df6['date'], df6['mortality rate'], color='steelblue')
#annotate max value
ymax = max(df6['mortality rate'])
xmax = df6.iloc[df6['mortality rate'].idxmax]['date']
plt.annotate(str(ymax), xy=(xmax, ymax), xytext=(-3,7), textcoords='offset points', color="r")
#for a,b in zip(df3['date'], df3['cases_daily']): 
#    plt.annotate(str(b), xy=(a, b), xytext=(-3,5), textcoords='offset points')

ax.set_title('Mortality rate over time', fontsize=20)
ax.set_ylabel('Mortality Rate', fontsize=15)
ax.set_xlabel('Date', fontsize=15)
#draw vertical line for every new months
feb = df6[df6['date']=='2020-02-01'].index.values.astype(int)[0]
mar = df6[df6['date']=='2020-03-01'].index.values.astype(int)[0]
apr = df6[df6['date']=='2020-04-01'].index.values.astype(int)[0]
plt.axvline(x=feb, color='indianred')
plt.axvline(x=mar, color='indianred')
plt.axvline(x=apr, color='indianred')
ax.get_xticklabels()[feb].set_color("indianred")
ax.get_xticklabels()[mar].set_color("indianred")
ax.get_xticklabels()[apr].set_color("indianred")
#color xtick red
max_day_y = df6.loc[df6['mortality rate'].idxmax()]['date']
max_day_x = df6[df6['date']==max_day_y].index.values.astype(int)[0]
ax.get_xticklabels()[max_day_x].set_color("red")
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
#plt.grid(linewidth=0.5)