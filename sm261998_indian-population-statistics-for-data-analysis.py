import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')
#Population
df = pd.read_csv('../input/indian-population-stats-for-data-analysis/india-population-2020-06-22.csv',

                sep=r'\s*,\s*', engine='python', skiprows=15)

df.head()
df.dtypes
df = df.rename(columns={'Annual % Change':'Annual_percent_change'})

df = df.rename(columns={'Population':'Population'})

df = df.rename(columns={'date':'Year'})
df['Population'] = df['Population'].apply(lambda x:"{:,}".format(x))
df['Year'] = df['Year'].str.split('-').str[0]

df['Year'] = df['Year'].astype(int)

df.head()
df.isnull().sum()
df['Annual_percent_change'] = df['Annual_percent_change'].fillna(0)
df_pop = df.iloc[:71]

year_list = df_pop['Year'].tolist()
plt.figure(figsize=(8,5))

plt.bar(df_pop['Year'][59:70],df_pop['Population'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Population')

plt.title('Population from 2009 to 2019')
plt.figure(figsize=(8,5))

plt.plot(df_pop['Year'][59:70],df_pop['Annual_percent_change'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Annual Percent Change')

plt.title('Annual change rate from 2009 to 2019')
plt.figure(figsize=(8,5))

plt.bar(df_pop['Year'][49:60],df_pop['Population'][49:60])

plt.xticks([1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])

plt.xlabel('Year')

plt.ylabel('Population')

plt.title('Population from 1999 to 2009')
# comparing Birth rate with previous 10 years

plt.figure(figsize=(12,6))

data_2000 = ('1,038,058,156',

 '1,056,575,549',

 '1,075,000,085',

 '1,093,317,189',

 '1,111,523,144',

 '1,129,623,456',

 '1,147,609,927',

 '1,165,486,291',

 '1,183,209,472',

 '1,200,669,765',

 '1,217,726,215')

data_2010 = ('1,217,726,215',

 '1,234,281,170',

 '1,250,287,943',

 '1,265,780,247',

 '1,280,842,125',

 '1,295,600,772',

 '1,310,152,403',

 '1,324,517,249',

 '1,338,676,785',

 '1,352,642,280',

 '1,366,417,754')

ind = np.arange(11)

width = 0.40

plt.bar(ind, data_2000, width, label = '2000')

plt.bar(ind+width, data_2010, width, label = '2010')

plt.xticks(ind+width/2, ('99/09','00/10','01/11','02/12','03/13','04/14','05/15','06/16','07/17','08/18','09/19'))

plt.title('Comparing Population with previous 10 years')

plt.xlabel('Years')

plt.ylabel('Number of People')

plt.legend()

plt.tight_layout()
df_p_cal = df_pop['Population'].str.replace(',','').astype(int)
print(df_p_cal[49:60].sum())

print(df_p_cal[59:70].sum())
df = pd.read_csv('../input/indian-population-stats-for-data-analysis/india-population-cbr.csv',

                sep=r'\s*,\s*', engine='python')

df.head(5)
df.dtypes
df = df.rename(columns={'Annual % Change':'Annual_percent_change'})

df = df.rename(columns={'Births per 1000 People':'Births_per_1000'})

df = df.rename(columns={'date':'Year'})

df['Year'] = df['Year'].str.split('/').str[-1]

df['Year'] = df['Year'].astype(int)
df.dtypes
df.isnull().sum()
df['Annual_percent_change'] = df['Annual_percent_change'].fillna(0)
df_cbr = df.iloc[:71]

year_list = df_cbr['Year'].tolist()
plt.figure(figsize=(8,5))

plt.plot(df_cbr['Year'][59:70],df_cbr['Births_per_1000'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

#plt.yticks(np.arange(17,23,step=0.5))

plt.xlabel('Year')

plt.ylabel('Births_per_1000')

plt.title('Birth rate from 2009 to 2019')
plt.figure(figsize=(8,5))

plt.plot(df_cbr['Year'][59:70],df_cbr['Annual_percent_change'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Annual Percent Change')

plt.title('Annual change rate from 2009 to 2019')
df_plot = df_cbr[['Births_per_1000','Annual_percent_change']][59:70]

ax = df_plot.plot(kind='line', figsize=(9,5))

ax.set_xticks(df_cbr.index[59:70]);

ax.set_xticklabels([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]);

ax.set_xlabel('Year')

ax.set_ylabel('Values')
plt.figure(figsize=(8,5))

plt.plot(df_cbr['Year'][49:60],df_cbr['Births_per_1000'][49:60])

plt.xticks([1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])

plt.xlabel('Year')

plt.ylabel('Births_per_1000')

plt.title('Birth rate from 1999 to 2009')
# comparing Birth rate with previous 10 years

x = [i for i in range(49,70,1)]

y = [i for i in range(1999,2020,1)]

plt.figure(figsize=(15,6))

plt.plot(df_cbr['Births_per_1000'][59:70],'--r', label='2009 to 2019')

plt.plot(df_cbr['Births_per_1000'][49:60],'o',label='1999 to 2009')

plt.xticks(x,y)

plt.xlabel('Year')

plt.ylabel('Births_per_1000')

plt.title('Birth rate comparison from 1999-09 to 2009-19')

plt.legend()
df = pd.read_csv('../input/indian-population-stats-for-data-analysis/india-population-death_rate.csv',

                sep=r'\s*,\s*', engine='python', skiprows=15)

df.head()
df = df.rename(columns={'Annual % Change':'Annual_percent_change'})

df = df.rename(columns={'Deaths per 1000 People':'Deaths_per_1000'})

df = df.rename(columns={'date':'Year'})
df.dtypes
df['Year'] = df['Year'].str.split('-').str[0]

df['Year'] = df['Year'].astype(int)
df.isnull().sum()
df['Annual_percent_change'] = df['Annual_percent_change'].fillna(0)

df['Year'].fillna(0, inplace=True)
df_dr = df.iloc[:71]

year_list = df_dr['Year'].tolist()

df_dr.head()
df.dtypes
plt.figure(figsize=(9,5))

plt.plot(df_dr['Year'][59:70],df_dr['Deaths_per_1000'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Deaths_per_1000')

plt.title('Death rate from 2009 to 2019')
plt.figure(figsize=(9,5))

plt.plot(df_dr['Year'][59:70],df_dr['Annual_percent_change'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Deaths_per_1000')

plt.title('Annual Rate from 2009 to 2019')
df_plot = df_dr[['Deaths_per_1000','Annual_percent_change']][59:70]

ax = df_plot.plot(kind='line', figsize=(9,5))

ax.set_xticks(df_dr.index[59:70]);

ax.set_xticklabels([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]);

ax.set_xlabel('Year')

ax.set_ylabel('Values')

ax.set_title('Death vs Annual percent change')
plt.figure(figsize=(9,5))

plt.plot(df_dr['Year'][49:60],df_dr['Deaths_per_1000'][49:60])

plt.xticks([1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])

plt.xlabel('Year')

plt.ylabel('Deaths_per_1000')

plt.title('Death rate from 1990 to 2009')
# comparing Death rate with previous 10 years

x = [i for i in range(49,71,1)]

y = [i for i in range(1999,2020,1)]

plt.figure(figsize=(15,6))

plt.plot(df_dr['Deaths_per_1000'][59:70],'--r', label='2009 to 2019')

plt.plot(df_dr['Deaths_per_1000'][49:60],'o',label='1999 to 2009')

plt.xticks(x,y)

plt.xlabel('Year')

plt.ylabel('Deaths_per_1000')

plt.legend()

plt.title('Death rate comparison from 1990-09 to 2009-19')
df = pd.read_csv('../input/indian-population-stats-for-data-analysis/india-population-fertitltyrate.csv',

                sep=r'\s*,\s*', engine='python', skiprows=15)

df.head()
df = df.rename(columns={'Annual % Change':'Annual_percent_change'})

df = df.rename(columns={'Births per Woman':'Births_per_woman'})

df = df.rename(columns={'date':'Year'})
df.dtypes
df['Year'] = df['Year'].str.split('-').str[0]

df['Year'] = df['Year'].astype(int)
df.isnull().sum()
df['Annual_percent_change'] = df['Annual_percent_change'].fillna(0)
df_fr = df.iloc[:71]

df_fr.head()
plt.figure(figsize=(9,5))

plt.plot(df_fr['Year'][59:70],df_fr['Births_per_woman'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Births_per_woman')

plt.title('Births per woman from 2009 to 2019')
plt.figure(figsize=(9,5))

plt.plot(df_dr['Year'][59:70],df_dr['Annual_percent_change'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Annual_percent_change')

plt.title('Annual Rate from 2009 to 2019')
df_plot = df_fr[['Births_per_woman','Annual_percent_change']][59:70]

ax = df_plot.plot(kind='line', figsize=(9,5))

ax.set_xticks(df_fr.index[59:70]);

ax.set_xticklabels([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]);

ax.set_xlabel('Year')

ax.set_ylabel('Values')

ax.set_title('Birth vs Annual percent change')
plt.figure(figsize=(9,5))

plt.plot(df_fr['Year'][49:60],df_fr['Births_per_woman'][49:60])

plt.xticks([1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])

plt.xlabel('Year')

plt.ylabel('Births_per_woman')

plt.title('Births_per_woman from 1990 to 2009')
# comparing Death rate with previous 10 years

x = [i for i in range(49,71,1)]

y = [i for i in range(1999,2020,1)]

plt.figure(figsize=(15,6))

plt.plot(df_fr['Births_per_woman'][59:70],'--r', label='2009 to 2019')

plt.plot(df_fr['Births_per_woman'][49:60],'o',label='1999 to 2009')

plt.xticks(x,y)

plt.xlabel('Year')

plt.ylabel('Births_per_woman')

plt.legend()

plt.title('Births_per_woman comparison from 1990-09 to 2009-19')
df = pd.read_csv('../input/indian-population-stats-for-data-analysis/india-population-infantmr.csv',

                sep=r'\s*,\s*', engine='python', skiprows=15)

df.head()
df = df.rename(columns={'Annual % Change':'Annual_percent_change'})

df = df.rename(columns={'Deaths per 1000 Live Births':'Deaths_per_1000_live_births'})

df = df.rename(columns={'date':'Year'})
df.dtypes
df['Year'] = df['Year'].str.split('-').str[0]

df['Year'] = df['Year'].astype(int)
df.isnull().sum()
df['Annual_percent_change'] = df['Annual_percent_change'].fillna(0)
df_imr = df.iloc[:71]

df_imr.head()
plt.figure(figsize=(9,5))

plt.plot(df_imr['Year'][59:70],df_imr['Deaths_per_1000_live_births'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Deaths_per_1000_live_births')

plt.title('Deaths_per_1000_live_births from 2009 to 2019')
plt.figure(figsize=(9,5))

plt.plot(df_imr['Year'][59:70],df_imr['Annual_percent_change'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Annual_percent_change')

plt.title('Annual Rate from 2009 to 2019')
df_plot = df_imr[['Deaths_per_1000_live_births','Annual_percent_change']][59:70]

ax = df_plot.plot(kind='line', figsize=(9,5))

ax.set_xticks(df_imr.index[59:70]);

ax.set_xticklabels([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]);

ax.set_xlabel('Year')

ax.set_ylabel('Values')

ax.set_title('Deaths_per_1000_live_births vs Annual percent change')
plt.figure(figsize=(9,5))

plt.plot(df_imr['Year'][49:60],df_imr['Deaths_per_1000_live_births'][49:60])

plt.xticks([1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])

plt.xlabel('Year')

plt.ylabel('Deaths_per_1000_live_births')

plt.title('Deaths_per_1000_live_births from 1990 to 2009')
# comparing Death rate with previous 10 years

x = [i for i in range(49,71,1)]

y = [i for i in range(1999,2020,1)]

plt.figure(figsize=(15,6))

plt.plot(df_imr['Deaths_per_1000_live_births'][59:70],'--r', label='2009 to 2019')

plt.plot(df_imr['Deaths_per_1000_live_births'][49:60],'o',label='1999 to 2009')

plt.xticks(x,y)

plt.xlabel('Year')

plt.ylabel('Deaths_per_1000_live_births')

plt.legend()

plt.title('Deaths_per_1000_live_births comparison from 1990-09 to 2009-19')
df = pd.read_csv('../input/indian-population-stats-for-data-analysis/india-population-lifeexp.csv',

                sep=r'\s*,\s*', engine='python', skiprows=15)

df.head()
df = df.rename(columns={'Annual % Change':'Annual_percent_change'})

df = df.rename(columns={'Life Expectancy from Birth (Years)':'Life_expectancy'})

df = df.rename(columns={'date':'Year'})
df.dtypes
df['Year'] = df['Year'].str.split('-').str[0]

df['Year'] = df['Year'].astype(int)
df.isnull().sum()
df['Annual_percent_change'] = df['Annual_percent_change'].fillna(0)

df_lf = df.iloc[:71]

df_lf.head()
plt.figure(figsize=(9,5))

plt.plot(df_lf['Year'][59:70],df_lf['Life_expectancy'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Life expectancy')

plt.title('Life Expectancy from 2009 to 2019')

plt.figure(figsize=(9,5))

plt.plot(df_lf['Year'][59:70],df_lf['Annual_percent_change'][59:70])

plt.xticks([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

plt.xlabel('Year')

plt.ylabel('Annual_percent_change')

plt.title('Annual Rate from 2009 to 2019')



df_plot = df_lf[['Life_expectancy','Annual_percent_change']][59:70]

ax = df_plot.plot(kind='line', figsize=(9,5))

ax.set_xticks(df_imr.index[59:70]);

ax.set_xticklabels([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]);

ax.set_xlabel('Year')

ax.set_ylabel('Values')

ax.set_title('Life Expectancy vs Annual percent change')
plt.figure(figsize=(9,5))

plt.plot(df_lf['Year'][49:60],df_lf['Life_expectancy'][49:60])

plt.xticks([1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])

plt.xlabel('Year')

plt.ylabel('Life Expectancy')

plt.title('Life Expectancy from 1990 to 2009')

plt.xlabel('Year')

# comparing Life expectancy rate with previous 10 years

x = [i for i in range(49,71,1)]

y = [i for i in range(1999,2020,1)]

plt.figure(figsize=(15,6))

plt.plot(df_lf['Life_expectancy'][59:70],'--r', label='2009 to 2019')

plt.plot(df_lf['Life_expectancy'][49:60],'o',label='1999 to 2009')

plt.xticks(x,y)

plt.xlabel('Year')

plt.ylabel(' Life expectancy')

plt.legend()

plt.title(' Life expectancy comparison from 1990-09 to 2009-19')
df = pd.read_csv('../input/indian-population-stats-for-data-analysis/india-suicide-rate.csv',

                sep=r'\s*,\s*', engine='python', skiprows=16)

df
df = df.drop(df.index[4])

df
df = df.rename(columns={'date':'Year'})

df = df.rename(columns={'Total':'Average'})

df = df.rename(columns={'Male':'Male'})

df = df.rename(columns={'Female':'Female'})
df.shape
df.isnull().sum()
df['Year'] = df['Year'].str.split('/').str[-1]

df['Year'] = df['Year'].astype(int)
df
plt.figure(figsize=(9,5))

plt.plot(df['Year'],df['Male'])

plt.xticks([2000,2005,2010,2015,2019])

plt.xlabel('Year')

plt.ylim([16,28])

plt.ylabel('Male Suicide rate per 1,00,000')

plt.title('Male Suicide rate till 2019')
plt.figure(figsize=(9,5))

plt.plot(df['Year'],df['Female'])

plt.xticks([2000,2005,2010,2015,2019])

plt.xlabel('Year')

plt.ylim([14,18])

plt.ylabel('Female Suicide rate per 1,00,000')

plt.title('Female Suicide rate till 2019')
plt.figure(figsize=(9,5))

plt.plot(df['Year'],df['Average'])

plt.xticks([2000,2005,2010,2015,2019])

plt.ylim([15,22])

#plt.xlim([1999,2022])

plt.xlabel('Year')

plt.ylabel('Average Suicide rate per 1,00,000')

plt.title('Average Suicide rate till 2019')
f = []

m = []

y=[]

for i in df['Male']:

    m.append(i)

for i in df['Female']:

    f.append(i)

for i in df['Year']:

    y.append(i)

f = tuple(f)

m = tuple(m)

y = tuple(y)


# comparing suicide rate with Male vs Female

plt.figure(figsize=(15,6))

ax = plt.plot(m,'--r', label='Male')

ax2 = plt.plot(f,'--b',label='Female')



#plt.ylim([12,28])

plt.xticks(np.arange(6), ['2000','2005', '2010', '2015', '2016', '2019'])

plt.xlabel('Year')

plt.ylabel('Values')

plt.legend()

plt.title('Male vs Female Suicide rate')
df