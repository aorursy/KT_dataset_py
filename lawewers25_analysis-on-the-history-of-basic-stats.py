%matplotlib inline
import pandas as pd

import matplotlib as plt

from pylab import *

import seaborn as sns

from sqlalchemy import create_engine



# Create connection.

engine = create_engine('sqlite:///:memory:')
batting = pd.DataFrame.from_csv('../input/batting.csv', index_col = None, encoding = 'utf-8')

batting.head()
#Load pitching data

pitching = pd.DataFrame.from_csv('../input/pitching.csv', index_col = None, encoding = 'utf-8')

pitching.head()
batting.to_sql('batting', engine, index = False)
pd.read_sql_table('batting', engine).head()
pitching.to_sql('pitching', engine, index = False)

pd.read_sql_table('pitching', engine).head()
#Queries to pull data for league wide HR and AVG by year

total_hr = pd.read_sql_query('SELECT year, SUM(hr) AS total_hr FROM batting GROUP BY year', engine)

total_avg = pd.read_sql_query('SELECT year, SUM(ab) AS total_ab, SUM(h) AS total_h FROM batting GROUP BY year', engine)
#Adding a column to calculate league wide batting average.

ab = total_avg['total_ab']

h = total_avg['total_h']

total_avg['avg'] = (h / ab)
year = total_hr['year']

year.astype('int')

hr = total_hr['total_hr']

year_avg = total_avg['year']

avg = total_avg['avg']
#Query to pull total league strikeouts

df_so = pd.read_sql_query('SELECT year, SUM(so) AS so FROM pitching GROUP BY year', engine)

year_so = df_so['year']

so = df_so['so']
#Query to pull data for ERA

df_era = pd.read_sql_query('SELECT year, SUM(er) AS total_er, SUM(ipouts) / 3 as total_ip FROM pitching GROUP BY year', engine)

#Calculations for league wide ERA

df_era['yr_era'] = (df_era['total_er'] / df_era['total_ip']) * 9

year_era = df_era['year']

era = df_era['yr_era']

fig = plt.figure(figsize=(8,4), dpi=100)
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15, 15), sharex = True)



ax1.bar(year, hr, align = 'center', width = .7, alpha = .5, color = 'red')

ax1.set_xlim([1871,2016])

ax1.set_xlabel('Year')

ax1.set_ylabel('Total Home Runs Hit')

ax1.set_title('Home Run History')



ax2.bar(year_so, so, align = 'center', width = .7, alpha = .5, color = 'blue')

ax2.set_xlim([1871,2016])

ax2.set_xlabel('Year')

ax2.set_ylabel('Total Strikeouts')

ax2.set_title('Strikeout History')



for x in year :

    #19th Century

    ax1.axvline(x=1900.5,c="black",linewidth=.5)

    ax2.axvline(x=1900.5,c="black",linewidth=.5)

    #Dead Ball

    ax1.axvline(x=1919.5,c="black",linewidth=.5)

    ax2.axvline(x=1919.5,c="black",linewidth=.5)

    #Lively Ball

    ax1.axvline(x=1940.5,c="black",linewidth=.5)

    ax2.axvline(x=1940.5,c="black",linewidth=.5)    

    #Integration

    ax1.axvline(x=1960.5,c="black",linewidth=.5)

    ax2.axvline(x=1960.5,c="black",linewidth=.5)

    #Expansion

    ax1.axvline(x=1976.5,c="black",linewidth=.5)

    ax2.axvline(x=1976.5,c="black",linewidth=.5) 

    #Free Agency

    ax1.axvline(x=1993.5,c="black",linewidth=.5)

    ax2.axvline(x=1993.5,c="black",linewidth=.5)

    #Steroid

    ax1.axvline(x=2005.5,c="black",linewidth=.5)

    ax2.axvline(x=2005.5,c="black",linewidth=.5)



ax1.text(1880, -575, '19th Century', fontsize = 12, color = 'black')

ax1.text(1905, -575, 'Dead Ball', fontsize = 12, color = 'black')

ax1.text(1925, -575, 'Lively Ball', fontsize = 12, color = 'black')

ax1.text(1946, -575, 'Integration', fontsize = 12, color = 'black')

ax1.text(1964, -575, 'Expansion', fontsize = 12, color = 'black')

ax1.text(1979, -575, 'Free Agency', fontsize = 12, color = 'black')

ax1.text(1996, -575, 'Steroid', fontsize = 12, color = 'black')

ax1.text(2007, -575, 'Modern', fontsize = 12, color = 'black')

ax1.text(1862, -575, 'Eras', fontsize = 15, color = 'black')

None
hr_so = pd.merge(total_hr, df_so, on = 'year')



fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 8), sharex = True)



sns.regplot(x="year", y="total_hr", truncate=False, data=hr_so, ax=ax1, color='red')

ax1.set_xlabel('Year')

ax1.set_ylabel('Home Runs')

ax1.set_title('League Home Run Totals')

ax1.set_ylim([0,6000])



sns.regplot(x="year", y="so", truncate=False, data=hr_so, ax=ax2, color='blue')

ax2.set_xlabel('Year')

ax2.set_ylabel('Strikeouts')

ax2.set_title('League Strikeout Totals')

ax2.set_ylim([0,40000])

None
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15, 15), sharex = True)



ax1.bar(year_avg, avg, align = 'center', width = .7, alpha = .5, color = 'red')

ax1.set_xlim([1871,2016])

ax1.set_xlabel('Year')

ax1.set_ylabel('League Wide Batting Average')

ax1.set_title('Batting Average History')



ax2.bar(year_era, era, align = 'center', width = .7, alpha = .5, color = 'blue')

ax2.set_xlim([1871,2016])

ax2.set_xlabel('Year')

ax2.set_ylabel('League Wide Earned Run Average')

ax2.set_title('ERA History')



for x in year :

    #19th Century

    ax1.axvline(x=1900.5,c="black",linewidth=.5)

    ax2.axvline(x=1900.5,c="black",linewidth=.5)

    #Dead Ball

    ax1.axvline(x=1919.5,c="black",linewidth=.5)

    ax2.axvline(x=1919.5,c="black",linewidth=.5)

    #Lively Ball

    ax1.axvline(x=1940.5,c="black",linewidth=.5)

    ax2.axvline(x=1940.5,c="black",linewidth=.5)    

    #Integration

    ax1.axvline(x=1960.5,c="black",linewidth=.5)

    ax2.axvline(x=1960.5,c="black",linewidth=.5)

    #Expansion

    ax1.axvline(x=1976.5,c="black",linewidth=.5)

    ax2.axvline(x=1976.5,c="black",linewidth=.5) 

    #Free Agency

    ax1.axvline(x=1993.5,c="black",linewidth=.5)

    ax2.axvline(x=1993.5,c="black",linewidth=.5)

    #Steroid

    ax1.axvline(x=2005.5,c="black",linewidth=.5)

    ax2.axvline(x=2005.5,c="black",linewidth=.5)



ax1.text(1875, -.03, '19th Century', fontsize = 12, color = 'black')

ax1.text(1902, -.03, 'Dead Ball', fontsize = 12, color = 'black')

ax1.text(1926, -.03, 'Lively Ball', fontsize = 12, color = 'black')

ax1.text(1943, -.03, 'Integration', fontsize = 12, color = 'black')

ax1.text(1963, -.03, 'Expansion', fontsize = 12, color = 'black')

ax1.text(1978, -.03, 'Free Agency', fontsize = 12, color = 'black')

ax1.text(1995, -.03, 'Steroid', fontsize = 12, color = 'black')

ax1.text(2007, -.03, 'Modern', fontsize = 12, color = 'black')

ax1.text(1862, -.03, 'Eras', fontsize = 15, color = 'black')

None
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 8), sharex = True)



sns.regplot(x="year", y="avg", truncate=False, data=total_avg, ax=ax1, color='red')

ax1.set_xlabel('Year')

ax1.set_ylabel('Batting Average')

ax1.set_title('League Batting Average')



sns.regplot(x="year", y="yr_era", truncate=False, data=df_era, ax=ax2, color='blue')

ax2.set_xlabel('Year')

ax2.set_ylabel('Earned Run Average')

ax2.set_title('League Earned Run Average')

None
exp_hr = total_hr.loc[total_hr['year'] >= 1961, :].reset_index()

exp_avg = total_avg.loc[total_avg['year'] >= 1961, :].reset_index()

exp_so = df_so.loc[df_so['year'] >= 1961, :].reset_index()

exp_era = df_era.loc[df_era['year'] >= 1961, :].reset_index()

exp_era.head()
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15, 15), sharex = True)



ax1.bar(exp_hr['year'], exp_hr['total_hr'], align = 'center', width = .7, alpha = .5, color = 'red')

ax1.set_xlim([1960,2016])

ax1.set_xlabel('Year')

ax1.set_ylabel('League Home Runs')

ax1.set_title('Home Run History After Expansion')



ax2.bar(exp_so['year'], exp_so['so'], align = 'center', width = .7, alpha = .5, color = 'blue')

ax2.set_xlim([1960,2016])

ax2.set_xlabel('Year')

ax2.set_ylabel('League Wide Stirkeouts')

ax2.set_title('Strikeout History after Expansion')

None
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15, 15), sharex = True)



ax1.bar(exp_avg['year'], exp_avg['avg'], align = 'center', width = .7, alpha = .5, color = 'red')

ax1.set_xlim([1960,2016])

ax1.set_xlabel('Year')

ax1.set_ylabel('League Wide Batting Average')

ax1.set_title('Batting Average History After Expansion Era')



ax2.bar(exp_era['year'], exp_era['yr_era'], align = 'center', width = .7, alpha = .5, color = 'blue')

ax2.set_xlim([1960,2016])

ax2.set_xlabel('Year')

ax2.set_ylabel('League Wide Earned Run Average')

ax2.set_title('ERA History After Expansion Era')

None
fig, ax = plt.subplots(2,2, figsize = (15, 8), sharex = True)



sns.regplot(x="year", y="total_hr", truncate=False, data=exp_hr, ax=ax[0,0], color='red')

ax[0,0].set_xlabel('Year')

ax[0,0].set_ylabel('Home Runs')

ax[0,0].set_title('League Home Run Totals After Expansion Era')

ax[0,0].set_ylim([0,6000])



sns.regplot(x="year", y="so", truncate=False, data=exp_so, ax=ax[1,0], color='blue')

ax[1,0].set_xlabel('Year')

ax[1,0].set_ylabel('Strikeouts')

ax[1,0].set_title('League Strikeout Totals After Expansion Era')

ax[1,0].set_ylim([0,40000])



sns.regplot(x="year", y="avg", truncate=False, data=exp_avg, ax=ax[0,1], color='red')

ax[0,1].set_xlabel('Year')

ax[0,1].set_ylabel('Batting Average')

ax[0,1].set_title('League Batting Average After Expansion Era')



sns.regplot(x="year", y="yr_era", truncate=False, data=exp_era, ax=ax[1,1], color='blue')

ax[1,1].set_xlabel('Year')

ax[1,1].set_ylabel('Earned Run Average')

ax[1,1].set_title('League Earned Run Average After Expansion Era')

None
fig = plt.figure(figsize=(10,10), dpi=100)
fig, ax = plt.subplots(2, 2, figsize = (15, 10), sharex = True)

modern_hr = total_hr.loc[year >= 2005, :]

modern_hr = modern_hr.reset_index()

modern_avg = total_avg.loc[year >= 2005, :].reset_index()

modern_so = df_so.loc[year >= 2005, :].reset_index()

modern_era = df_era.loc[year >= 2005, :].reset_index()



ax[0,0].bar(modern_hr['year'], modern_hr['total_hr'], align = 'center', width = .7, alpha = .5, color = 'red')

ax[0,0].set_ylabel('Total Home Runs')

ax[0,0].set_title('Home Runs')

ax[0,1].bar(modern_avg['year'], modern_avg['avg'], align = 'center', width = .7, alpha = .5, color = 'red')

ax[0,1].set_ylabel('League Batting Average')

ax[0,1].set_title('Batting Average')

ax[1,0].bar(modern_so['year'], modern_so['so'], align = 'center', width = .7, alpha = .5, color = 'blue')

ax[1,0].set_xlabel('Year')

ax[1,0].set_ylabel('Total Strikeouts')

ax[1,0].set_title('Strikeouts')

ax[1,1].bar(modern_era['year'], modern_era['yr_era'], align = 'center', width = .7, alpha = .5, color = 'blue')

ax[1,1].set_xlabel('Year')

ax[1,1].set_ylabel('League ERA')

ax[1,1].set_title('Earned Run Average')

None