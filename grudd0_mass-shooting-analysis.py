# import packages 

%matplotlib inline

import numpy as np

import pandas as pd

import datetime

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.dates as mdates

from sklearn import linear_model
# create dataframe with USA population data

# http://www.multpl.com/united-states-population/table

pop_dict = {'date': {pd.Timestamp('1966-01-01'): pd.Timestamp('1966-01-01'),

  pd.Timestamp('1967-01-01'): pd.Timestamp('1967-01-01'),

  pd.Timestamp('1968-01-01'): pd.Timestamp('1968-01-01'),

  pd.Timestamp('1969-01-01'): pd.Timestamp('1969-01-01'),

  pd.Timestamp('1970-01-01'): pd.Timestamp('1970-01-01'),

  pd.Timestamp('1971-01-01'): pd.Timestamp('1971-01-01'),

  pd.Timestamp('1972-01-01'): pd.Timestamp('1972-01-01'),

  pd.Timestamp('1973-01-01'): pd.Timestamp('1973-01-01'),

  pd.Timestamp('1974-01-01'): pd.Timestamp('1974-01-01'),

  pd.Timestamp('1975-01-01'): pd.Timestamp('1975-01-01'),

  pd.Timestamp('1976-01-01'): pd.Timestamp('1976-01-01'),

  pd.Timestamp('1977-01-01'): pd.Timestamp('1977-01-01'),

  pd.Timestamp('1978-01-01'): pd.Timestamp('1978-01-01'),

  pd.Timestamp('1979-01-01'): pd.Timestamp('1979-01-01'),

  pd.Timestamp('1980-01-01'): pd.Timestamp('1980-01-01'),

  pd.Timestamp('1981-01-01'): pd.Timestamp('1981-01-01'),

  pd.Timestamp('1982-01-01'): pd.Timestamp('1982-01-01'),

  pd.Timestamp('1983-01-01'): pd.Timestamp('1983-01-01'),

  pd.Timestamp('1984-01-01'): pd.Timestamp('1984-01-01'),

  pd.Timestamp('1985-01-01'): pd.Timestamp('1985-01-01'),

  pd.Timestamp('1986-01-01'): pd.Timestamp('1986-01-01'),

  pd.Timestamp('1987-01-01'): pd.Timestamp('1987-01-01'),

  pd.Timestamp('1988-01-01'): pd.Timestamp('1988-01-01'),

  pd.Timestamp('1989-01-01'): pd.Timestamp('1989-01-01'),

  pd.Timestamp('1990-01-01'): pd.Timestamp('1990-01-01'),

  pd.Timestamp('1991-01-01'): pd.Timestamp('1991-01-01'),

  pd.Timestamp('1992-01-01'): pd.Timestamp('1992-01-01'),

  pd.Timestamp('1993-01-01'): pd.Timestamp('1993-01-01'),

  pd.Timestamp('1994-01-01'): pd.Timestamp('1994-01-01'),

  pd.Timestamp('1995-01-01'): pd.Timestamp('1995-01-01'),

  pd.Timestamp('1996-01-01'): pd.Timestamp('1996-01-01'),

  pd.Timestamp('1997-01-01'): pd.Timestamp('1997-01-01'),

  pd.Timestamp('1998-01-01'): pd.Timestamp('1998-01-01'),

  pd.Timestamp('1999-01-01'): pd.Timestamp('1999-01-01'),

  pd.Timestamp('2000-01-01'): pd.Timestamp('2000-01-01'),

  pd.Timestamp('2001-01-01'): pd.Timestamp('2001-01-01'),

  pd.Timestamp('2002-01-01'): pd.Timestamp('2002-01-01'),

  pd.Timestamp('2003-01-01'): pd.Timestamp('2003-01-01'),

  pd.Timestamp('2004-01-01'): pd.Timestamp('2004-01-01'),

  pd.Timestamp('2005-01-01'): pd.Timestamp('2005-01-01'),

  pd.Timestamp('2006-01-01'): pd.Timestamp('2006-01-01'),

  pd.Timestamp('2007-01-01'): pd.Timestamp('2007-01-01'),

  pd.Timestamp('2008-01-01'): pd.Timestamp('2008-01-01'),

  pd.Timestamp('2009-01-01'): pd.Timestamp('2009-01-01'),

  pd.Timestamp('2010-01-01'): pd.Timestamp('2010-01-01'),

  pd.Timestamp('2011-01-01'): pd.Timestamp('2011-01-01'),

  pd.Timestamp('2012-01-01'): pd.Timestamp('2012-01-01'),

  pd.Timestamp('2013-01-01'): pd.Timestamp('2013-01-01'),

  pd.Timestamp('2014-01-01'): pd.Timestamp('2014-01-01'),

  pd.Timestamp('2015-01-01'): pd.Timestamp('2015-01-01'),

  pd.Timestamp('2016-01-01'): pd.Timestamp('2016-01-01'),

  pd.Timestamp('2017-01-01'): pd.Timestamp('2017-01-01')},

 'population': {pd.Timestamp('1966-01-01 00:00:00'): 196.56,

  pd.Timestamp('1967-01-01'): 198.7,

  pd.Timestamp('1968-01-01'): 200.7,

  pd.Timestamp('1969-01-01'): 202.6,

  pd.Timestamp('1970-01-01'): 205.0,

  pd.Timestamp('1971-01-01'): 207.66,

  pd.Timestamp('1972-01-01'): 209.9,

  pd.Timestamp('1973-01-01'): 211.9,

  pd.Timestamp('1974-01-01'): 213.8,

  pd.Timestamp('1975-01-01'): 215.9,

  pd.Timestamp('1976-01-01'): 218.0,

  pd.Timestamp('1977-01-01'): 220.2,

  pd.Timestamp('1978-01-01'): 222.5,

  pd.Timestamp('1979-01-01'): 225.0,

  pd.Timestamp('1980-01-01'): 227.2,

  pd.Timestamp('1981-01-01'): 229.4,

  pd.Timestamp('1982-01-01'): 231.6,

  pd.Timestamp('1983-01-01'): 233.7,

  pd.Timestamp('1984-01-01'): 235.8,

  pd.Timestamp('1985-01-01'): 237.9,

  pd.Timestamp('1986-01-01'): 240.1,

  pd.Timestamp('1987-01-01'): 242.2,

  pd.Timestamp('1988-01-01'): 244.5,

  pd.Timestamp('1989-01-01'): 246.8,

  pd.Timestamp('1990-01-01'): 249.6,

  pd.Timestamp('1991-01-01'): 252.9,

  pd.Timestamp('1992-01-01'): 256.5,

  pd.Timestamp('1993-01-01'): 259.9,

  pd.Timestamp('1994-01-01'): 263.1,

  pd.Timestamp('1995-01-01'): 266.2,

  pd.Timestamp('1996-01-01'): 269.3,

  pd.Timestamp('1997-01-01'): 272.6,

  pd.Timestamp('1998-01-01'): 275.8,

  pd.Timestamp('1999-01-01'): 279.0,

  pd.Timestamp('2000-01-01'): 282.1,

  pd.Timestamp('2001-01-01'): 284.9,

  pd.Timestamp('2002-01-01'): 287.6,

  pd.Timestamp('2003-01-01'): 290.1,

  pd.Timestamp('2004-01-01'): 292.8,

  pd.Timestamp('2005-01-01'): 295.5,

  pd.Timestamp('2006-01-01'): 298.3,

  pd.Timestamp('2007-01-01'): 301.2,

  pd.Timestamp('2008-01-01'): 304.0,

  pd.Timestamp('2009-01-01'): 306.7,

  pd.Timestamp('2010-01-01'): 309.3,

  pd.Timestamp('2011-01-01'): 311.6,

  pd.Timestamp('2012-01-01'): 314.0,

  pd.Timestamp('2013-01-01'): 316.1,

  pd.Timestamp('2014-01-01'): 318.5,

  pd.Timestamp('2015-01-01'): 320.8,

  pd.Timestamp('2016-01-01'): 323.1,

  pd.Timestamp('2017-01-01'): 325.3}}

df_pop = pd.DataFrame(pop_dict)

df_pop.columns = ['date','population']

df_pop.date = pd.to_datetime(df_pop.date)

# set dataframe index to Date colume

df_pop.index = df_pop.date
# read mass shootings dataset into pandas dataframe

df = pd.read_csv('../input/Mass Shootings Dataset Ver 5.csv',encoding = "ISO-8859-1")

# convert Date colume to datetime format

df.Date = pd.to_datetime(df.Date)

# set dataframe index to Date colume

df.index = df.Date

df.info()
# make a date series for ploting, freq = AS to start at left end of intervals

date = pd.date_range(start='1966-01-01', end='2017-01-01', freq='AS')

# ratio for figure size argument

ratio = (10,6)
# analysis for yearly attack counts

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

ax.plot(date,df.Fatalities.resample('AS').count().fillna(0),'-o')

# set xaxis major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# set grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('Year')

plt.ylabel('Number Attacks');

plt.title('Number Attacks by Year');
# attacks per million population

pop = df_pop.population

count = df.Fatalities.resample('AS').count().fillna(0)

count_pop = count / pop

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

date2 = pd.date_range(start='1966-01-01', end='2017-01-01', freq='AS')

ax.plot(date2,count_pop,'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('Year')

plt.ylabel('Attacks per Million');

plt.title('Attacks per Million People');
# Attacks Rolling Average 5 Year Windows

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

c = df.Fatalities.resample("AS").count().fillna(0).rolling(window=5, min_periods=5).sum()

ax.plot(c.index,c,'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('5 Year Window End Point')

plt.ylabel('Attacks');

plt.title('Attacks for Rolling 5 Year Window');
# Attacks per Year per Million Population for Rolling 5 Year Windows

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

#date2 = pd.date_range(start='1966-01-01', end='2017-01-01', freq='AS')

date2 = pd.date_range(start='1970-01-01', end='2021-01-01', freq='AS')

s = df.Fatalities.resample("AS").count().fillna(0).rolling(window=5, min_periods=5).mean()

p = df_pop.population.rolling(window=5, min_periods=5).mean()

sp = s/p

ax.plot(sp.index,s/p,'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('end of 5 Year segment')

plt.ylabel('Attacks per Year per Million Population');

plt.title('Attacks per Year per Million Population for Rolling 5 Year Windows');
# analysis for yearly fatalities

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

ax.plot(date,df.Fatalities.resample('AS').sum().fillna(0),'-o')

# set major xaxis labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# set grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('Year')

plt.ylabel('Fatalities');

plt.title('Fatalities per Year');
# analysis fatalities per shooting by year

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

ax.plot(date,df.Fatalities.resample('AS').mean().fillna(0),'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('Year')

plt.ylabel('Fatalities per Attack');

plt.title('Fatalities per Attack by Year');
# Fatalities per Year Rolling Average 5 Year Windows

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

f = df.Fatalities.resample("AS").sum().fillna(0).rolling(window=5, min_periods=5).sum()

c = df.Fatalities.resample("AS").count().fillna(0).rolling(window=5, min_periods=5).sum()

ax.plot(f.index,f/c,'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('5 Year Window End Point')

plt.ylabel('Fatalities per Shooting');

plt.title('Fatalities per Shooting for Rolling 5 Year Window');
# Fatalities per Million People in Population by Year

pop = df_pop.population

sum = df.Fatalities.resample('AS').sum().fillna(0)

sum_pop = sum / pop

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

date2 = pd.date_range(start='1966-01-01', end='2017-01-01', freq='AS')

ax.plot(date2,sum_pop,'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('Year')

plt.ylabel('Fatalities per Million');

plt.title('Fatalities per Million People in Population by Year');
# Fatalities per Year for Rolling 5 Year Windows

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

s = df.Fatalities.resample("AS").sum().fillna(0).rolling(window=5, min_periods=5).mean()

ax.plot(s.index,s,'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('end of 5 Year segment')

plt.ylabel('Fatalities per Year');

plt.title('Fatalities per Year for Rolling 5 Year Windows');
# Fatalities per Year for Rolling 5 Year Windows

plt.close('all')

fig,ax = plt.subplots(figsize=ratio)

#date2 = pd.date_range(start='1966-01-01', end='2017-01-01', freq='AS')

date2 = pd.date_range(start='1970-01-01', end='2021-01-01', freq='AS')

s = df.Fatalities.resample("AS").sum().fillna(0).rolling(window=5, min_periods=5).mean()

p = df_pop.population.rolling(window=5, min_periods=5).mean()

#ax.plot(date2,s/p,'-o')

ax.plot(s.index,s/p,'-o')

# major labels

ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# grid lines

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('end of 5 Year segment')

plt.ylabel('Fatalities per Year per Million Population');

plt.title('Fatalities per Year per Million Population for Rolling 5 Year Windows');
# get value counts for Race

df.Race.value_counts()
# edit Race colume to reduce number of categories

df.Race = df.Race.str.strip()

to_replace_list = ['white',

'White',

'Some other race',

'unclear',

'Unknown',

'White American or European American/Some other Race',

'Asian American',

'Asian American/Some other race',

'black',

'Black',

'Black American or African American/Unknown',

'Native American or Alaska Native'] 

replace_list = ['White American or European American',

'White American or European American',

'Other',

'Other',

'Other',

'White American or European American',

'Asian',

'Asian',

'Black American or African American',

'Black American or African American',

'Black American or African American',

'Native American']

df.Race = df.Race.replace(to_replace_list,replace_list)

race = df.Race.value_counts()

race
# plot pie chart for race distribution

fig,ax = plt.subplots(figsize=(10,10))

ax.pie(x=race,labels=race.index,rotatelabels=False, autopct='%.2f%%');

plt.title('Shooter Race Distribution');
# group race colume in 5 year segments

df_race = df.groupby(pd.Grouper(key='Date', freq='5AS'))['Race'].value_counts()

ticklabels = ['1966 - 1970','1971 - 1975','1976 - 1980','1981 - 1985','1986 - 1990','1991 - 1995','1996 - 2000',

'2001 - 2005','2006 - 2010','2011 - 2015','2016 - 2017']

df_race
# Shooter Race Distribution 5 Year Windows

df_race_us = df_race.unstack()

ax = df_race_us.plot(kind='bar',x=df_race_us.index,stacked=True,figsize=(10,6))

plt.title('Shooter Race Distribution 5 Year Window')

plt.xlabel('5 Year Window')

plt.ylabel('Shootings')

ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.gcf().autofmt_xdate()
# norm shooter race distribution

size = df.groupby(pd.Grouper(key='Date', freq='5AS'))['Race'].size()

norm = df_race / size * 100
# Normerilzed Shooter Race distribution 5 Year Windows

norm_us = norm.unstack()

#ax = norm_us.plot(kind='bar',x=df_race_us.index,stacked=True,figsize=(20,10))

ax = norm_us.plot(kind='bar',x=norm_us.index,stacked=True,figsize=(10,6))

plt.title('Normerilzed Shooter Race Distribution 5 Year Windows')

plt.xlabel('5 Year Window')

plt.ylabel('Normilized Shootings %')

ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.gcf().autofmt_xdate()
# creats dataframe with Race Demographics

# https://en.wikipedia.org/wiki/Historical_racial_and_ethnic_demographics_of_the_United_States#Historical_

# data_for_all_races_and_for_Hispanic_origin_(1610%E2%80%932010)

non_hispanic_white = pd.Series([153217498,169622593,180256366,188128296,194552774,196817552])

hispanic_any_race = pd.Series([5814784,8920940,14608673,22354059,35305818,50477594])

# assume 5,000,000 for missing data for years 1960,1970,1980,1990 for two_or_more_races

two_or_more_races = pd.Series([5000000,5000000,5000000,5000000,6826228,9009073])

Some_other_race = pd.Series([87606,230064,6758319,9804847,15359073,19107368])

asian_pacific_islander = pd.Series([980337,1526401,3500439,7273662,10641833,15214265])

American_indian_eskimo = pd.Series([551669,795110,1420400,1959234,2475956,2932248])

Black = pd.Series([18871831,22539362,26495025,29986060,34658190,38929319])

White = pd.Series([158831732,178119221,188371622,199686070,211460626,223553265])

hispanic_non_white = hispanic_any_race - (White - non_hispanic_white)

df_race_dem = pd.DataFrame({'non_hispanic_white':non_hispanic_white,

                        'hispanic_any_race':hispanic_any_race,

                        'two_or_more_races':two_or_more_races,

                        'Some_other_race':Some_other_race,

                       'asian_pacific_islander':asian_pacific_islander,

                       'American_indian_eskimo':American_indian_eskimo,

                       'Black':Black,

                       'White':White,'hispanic_non_white':hispanic_non_white})

df_race_dem.index = [1960,1970,1980,1990,2000,2010]

# creat dataframe with Race Demographics means for 5 year segments

def extract(df,col):

    y = df[col].values.reshape(-1,1)

    x = df.index.values.reshape(-1,1).astype(float)

    reg_white = linear_model.LinearRegression().fit(x,y)

    ans = reg_white.predict(np.array(np.arange(1966,2018,1)).reshape(-1,1)).tolist()

    flattened = [int(val) for sublist in ans for val in sublist]

    return flattened

non_hispanic_white = pd.Series(extract(df_race_dem,'non_hispanic_white'))

hispanic_non_white = pd.Series(extract(df_race_dem,'hispanic_non_white'))

American_indian_eskimo = pd.Series(extract(df_race_dem,'American_indian_eskimo'))

Black = pd.Series(extract(df_race_dem,'Black'))

hispanic_any_race = pd.Series(extract(df_race_dem,'hispanic_any_race'))

Some_other_race = pd.Series(extract(df_race_dem,'Some_other_race'))

White = pd.Series(extract(df_race_dem,'White'))

asian_pacific_islander = pd.Series(extract(df_race_dem,'asian_pacific_islander'))

two_or_more_races = pd.Series(extract(df_race_dem,'two_or_more_races'))

df_race_dem_reg = pd.DataFrame({'Asian':asian_pacific_islander,

                                'Black American or African American':Black,

                                'Latino':hispanic_non_white,

                                'Native American':American_indian_eskimo,

                                'Other':Some_other_race,

                                'White American or European American':White,

                                'Two or more races':two_or_more_races,

                                })

date = pd.date_range(start='1966-01-01', end='2017-01-01', freq='AS')

df_race_dem_reg = df_race_dem_reg.set_index(date)

df_race_dem_reg_5 = df_race_dem_reg.resample("5AS").mean()

df_race_dem_reg_5
df_race_norm = df_race.unstack() / df_race_dem_reg_5 * 100000

df_race_norm
# Normerilzed Shooter Race distribution 5 Year Windows per 100,000 in poulation

#ax = norm_us.plot(kind='bar',x=df_race_us.index,stacked=True,figsize=(20,10))

ax = df_race_norm.plot(kind='bar',x=df_race_norm.index,stacked=False,figsize=(15,9))

plt.title('Normerilzed Shooter Race Distribution per 100,000 in 5 Year Windows')

plt.xlabel('5 Year Window')

plt.ylabel('Shootings per 100,000')

ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.gcf().autofmt_xdate()
# edit Gender colume to reduce number of categories

df.Gender = df.Gender.str.strip()

to_replace_list = ['M','M/F'] 

replace_list = ['Male','Male/Female']

df.Gender = df.Gender.replace(to_replace_list,replace_list)

Gender = df.Gender.value_counts()

Gender
# plot pie chart for race distribution

fig,ax = plt.subplots(figsize=(10,10))

ax.pie(x=Gender,labels=Gender.index,rotatelabels=False, autopct='%.2f%%');

plt.title('Shooter Gender Distribution');
# calculate weekday name column

df['day_of_week'] = df['Date'].dt.weekday_name

day_of_week = df.day_of_week.value_counts()
# plot pie chart for day of week

fig,ax = plt.subplots(figsize=(10,10))

ax.pie(x=day_of_week,labels=day_of_week.index,rotatelabels=False, autopct='%.2f%%');

plt.title('Day of Week Distribution');
# calculate weekday name column

df['month_of_year']= df['Date'].dt.month

month_of_year = df.month_of_year.value_counts()
# plot pie chart for day of week

fig,ax = plt.subplots(figsize=(10,10))

ax.pie(x=month_of_year,labels=month_of_year.index,rotatelabels=False, autopct='%.2f%%');

plt.title('Month of Year Distribution');
# fatalities by month

f_month = df.groupby(pd.Grouper(key='month_of_year'))['Fatalities'].sum()

f_month_sum = f_month.sum()

f_month_sum_norm = f_month / f_month_sum * 100

f_month_sum_norm_sort = f_month_sum_norm.sort_index()

f_month
# Normerilzed month fatality distribution

ax = f_month_sum_norm_sort.plot(kind='bar',x=f_month_sum_norm_sort.index,stacked=True,figsize=(10,6))

plt.title('Normerilzed Monthly Fatality Distribution')

plt.xlabel('Month')

plt.ylabel('Normilized Fatalities %')

#ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.gcf().autofmt_xdate()
# norm shootings  distribution

size = df.groupby(pd.Grouper(key='Date', freq='5AS'))['month_of_year'].size()



# group month of year colume in 5 year segments

df_month_5 = df.groupby(pd.Grouper(key='Date', freq='5AS'))['month_of_year'].value_counts()

ticklabels = ['1966 - 1970','1971 - 1975','1976 - 1980','1981 - 1985','1986 - 1990','1991 - 1995','1996 - 2000',

'2001 - 2005','2006 - 2010','2011 - 2015','2016 - 2017']

df_month_5_n = df_month_5 / size * 100
# Number of shooting by month 5 Year Windows

df_month_5_n_us = df_month_5_n.unstack()

ax = df_month_5_n_us.plot(kind='bar',x=df_month_5_n_us.index,stacked=True,figsize=(10,6),colormap='Paired')

plt.title('Number Shootings Normerilzed Month Distribution 5 Year Window')

plt.xlabel('5 Year Window')

plt.ylabel('Number Shootings')

ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.gcf().autofmt_xdate()
# plot pie chart mental health distribution

mental = df['Mental Health Issues'].value_counts()

fig,ax = plt.subplots(figsize=(10,10))

ax.pie(x=mental,labels=mental.index,rotatelabels=False, autopct='%.2f%%');

plt.title('Mental Health Distribution');
# plot pie chart mental health distribution

cause = df['Cause'].value_counts()

fig,ax = plt.subplots(figsize=(10,10))

ax.pie(x=cause,labels=cause.index,rotatelabels=False, autopct='%.2f%%');

plt.title('Cause Distribution');
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white',stopwords=stopwords,max_words=100,

                      max_font_size=40).generate(str(df['Summary']))

plt.figure(figsize=(10,6))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('max_colwidth',-1)

students = df[df['Summary'].str.contains('student')][['Title','Summary']]

students
students.info()
print('{0:.1f}% of the summary blocks contain the word student'.format(len(students)/len(df)*100))
ages = df.Age.values

sum = []

for i in ages:

    if type(i) == float:

        continue

    s = i.split(',')

    sum.append(s)

flattened = [int(val) for sublist in sum for val in sublist]
fig,ax = plt.subplots(figsize=ratio)

ax.hist(flattened,bins=10)

ax.xaxis.grid(False,'minor') # turn off minor tic grid lines

ax.xaxis.grid(True,'major') # turn on major tic grid lines;

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.xlabel('Age')

plt.ylabel('Number of shooters');

plt.title('Histogram of Shooter Age');
# plot pie chart for targets

target = df.Target.value_counts()

df.Target.value_counts()[:20].plot(kind='barh',figsize=(10,6))

plt.xlabel('Counts')

plt.title('What are  the Targets');
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly.graph_objs import *

init_notebook_mode()



#df_f = df[df['Fatalities'] > 5]

df['text'] = 'Fatalities ' + (df['Fatalities']).astype(str) + '<br>' + (df['Location']).astype(str)

scale = 5

killed = [dict(

    type = 'scattergeo',

    locationmode = 'USA-states',

    lon = df['Longitude'],

    lat = df['Latitude'],

     text = df['text'],

     marker = dict(

        size = df['Fatalities']*scale,

        color = "rgb(0,116,217)",

        line = dict(width=0.5, color='rgb(40,40,40)'),

        sizemode = 'area'

    ),

    #name = ?

)]

    

layout = dict(

    title = 'Mass Shooting Fatalities',

    showlegend = False,

    geo = dict(

        scope='usa',

        projection=dict( type='albers usa' ),

        showland = True,

        landcolor = 'rgb(217, 217, 217)',

        subunitwidth=1,

        countrywidth=1,

        subunitcolor="rgb(255, 255, 255)",

        countrycolor="rgb(255, 255, 255)"

    )

)

fig = dict( data=killed, layout=layout )

iplot( fig)