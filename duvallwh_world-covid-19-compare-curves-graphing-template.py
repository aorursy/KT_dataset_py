import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import warnings



warnings.filterwarnings('ignore')

register_matplotlib_converters()
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv', parse_dates=['Date'])
train['Province/State'] = train['Province/State'].fillna('No Sub') # I will fill the NAs for the areas with no sub-region

train['area_full'] = train['Country/Region'] + '_' + train['Province/State'] # make a unique identifier for each sub-region
most_cases = train.groupby('area_full').max().sort_values('ConfirmedCases', ascending=False).iloc[:30]

_, ax = plt.subplots(figsize=(10,8))

sns.barplot(most_cases['ConfirmedCases'], most_cases.index, ax=ax)

ax.set_title('Countries with the Most Cases')
train.shape
print(f"There are { len(train['area_full'].unique())} unique areas within {len(train['Country/Region'].unique())} Country/Regions. The training data set starts on "\

      f"{train.Date.min().strftime('%m-%d-%Y')} and goes through {train.Date.max().strftime('%m-%d-%Y')}. The test set starts on {test.Date.min().strftime('%m-%d-%Y')}"\

      f" and goes through {test.Date.max().strftime('%m-%d-%Y')}.")
train[train.ConfirmedCases == train.ConfirmedCases.max()]['area_full'].iloc[0] # find Wuhan's sub-area
def compare_curves(loc1, loc2, df=train):

    """Compare one curve to another with offset.

       loc1: location with no offset

       loc2: location with offset

       df: train dataframe

       offset: # of days"""

    myFmt = mdates.DateFormatter('%m/%d/%Y')

    days = mdates.DayLocator(interval=4)



    df1 = train[train['area_full'] == loc1]

    df2 = train[train['area_full'] == loc2]

    fig, ax = plt.subplots(figsize=(16,8))

    

    cases_min = 25

    if df1.ConfirmedCases.min() > cases_min: cases_min = df1.ConfirmedCases.min()

    

    offset = df1.loc[df1['ConfirmedCases'] > cases_min, 'Date'].iloc[0] - df2.loc[df2['ConfirmedCases'] > cases_min, 'Date'].iloc[0]

    df2['Date'] = df2['Date'] + offset

    

    ax.plot(df1['Date'], df1['ConfirmedCases'], label=loc1.replace('_No Sub', ''))

    ax.plot(df2['Date'], df2['ConfirmedCases'], label=loc2.replace('_No Sub', ''))

    

    start = df1.loc[df1['ConfirmedCases'] > 0, 'Date'].iloc[0]

    

    ax.set_xlim(start, train['Date'].max()+pd.DateOffset(2))

    ax.set_ylim(0, df1['ConfirmedCases'].max()*1.3)

    ax.grid()

    ax.xaxis.set_major_locator(days)

    ax.xaxis.set_major_formatter(myFmt)

    

    title_string = f"Comparison between Current {loc1.replace('_No Sub', '')} and {loc2.replace('_No Sub', '')} {offset.days} Days Ago"

    

    ax.set_title(title_string, size=20)

    ax.legend()

    fig.autofmt_xdate()



compare_curves('Spain_No Sub', 'Italy_No Sub', train)
compare_curves('Italy_No Sub', 'China_Hubei')
compare_curves('US_New York', 'China_Hubei')
compare_curves('US_Florida', 'US_New York')
compare_curves('US_California', 'US_Washington')
compare_curves('US_New York', 'Italy_No Sub')
def graph_cases(ax=False, fig=None, location=None, df=None, cases=True, deaths=True):

    df = df[df['area_full'] == location]

    location = location.replace('_No Sub', '') # get rid of No Sub if there is no subregion

    myFmt = mdates.DateFormatter('%m/%d/%Y')

    days = mdates.DayLocator(interval=4)



    if ax == False: 

        fig, ax = plt.subplots(figsize=(16,8))

    if cases: ax.plot(df.Date, df.ConfirmedCases, label=f'{location} Cases')

    if deaths: ax.plot(df.Date, df.Fatalities, label=f'{location} Fatalities')

    ax.xaxis.set_major_locator(days)

    ax.xaxis.set_major_formatter(myFmt)

    ax.grid()

    ax.legend()

    ax.set_title(f'COVID-19 Cases', size=26)

    fig.autofmt_xdate()

    return ax, fig
ax, fig = graph_cases(location='China_Hubei', df=train)

ax, fig = graph_cases(ax=ax, fig=fig, location='Italy_No Sub', df=train)

ax.plot([pd.datetime(2020,1,23), pd.datetime(2020,1,23)], [0,70000], 'g--', label='Hubei Lockdown Starts') ## turn info a function

ax.plot([pd.datetime(2020,3,10), pd.datetime(2020,3,10)], [0,70000], 'r--', label='Italy Locks down Country')

plt.grid()

ax.legend(loc='upper left')

plt.show()
ax, fig = graph_cases(location='Italy_No Sub', df=train)

# ax, fig = graph_cases(ax=ax, fig=fig, location='US_Washington', df=train)

ax, fig = graph_cases(ax=ax, fig=fig, location='France_France', df=train)

ax, fig = graph_cases(ax=ax, fig=fig, location='Iran_No Sub', df=train)

# ax.grid()

# ax.grid()

ax.set_xlim(pd.datetime(2020,2,15))
print(f"The number of cases in Hubei (Wuhan) when lockdown procedures were initiated: " \

      f"{ train[ (train['area_full'] == 'China_Hubei') & (train['Date'] == pd.datetime(2020,1,25))]['ConfirmedCases'].iloc[0]}")
START_DATE = pd.datetime(2020,1,25)



c_regions = china_regions = train[train['Country/Region'] == 'China']['area_full'].unique()



ax, fig = graph_cases(location='China_Anhui', df=train, deaths=False)

for reg in c_regions:

    if reg not in ['China_Anhui', 'China_Hubei']:

        ax, fig = graph_cases(ax=ax, fig=fig, location=reg, df=train, deaths=False)

        

ax.plot([START_DATE, START_DATE], [0,1400], 'g--', label='Start Date') ## turn info a function

ax.legend(bbox_to_anchor=(1.25,1)) # move the legend so you can see cases

ax.grid()
china = train[train['Country/Region'] == 'China']

num_shutdown = china[china['Date'] == START_DATE] # This number can be changed 

num_latest = china[china['Date'] == china['Date'].max()] # find the latest variables

sd = num_shutdown[['Province/State', 'ConfirmedCases']].merge(num_latest[['Province/State', 'ConfirmedCases']], on='Province/State', suffixes=('_sd', '_ct'))

sd['PercentGrowth'] = sd['ConfirmedCases_ct'] / sd['ConfirmedCases_sd']

sd = sd[sd['ConfirmedCases_sd'] > 0] # get rid of the infinity cases
_, axs = plt.subplots(ncols=3, figsize=(24,8))

axs[0].hist(sd['ConfirmedCases_sd'], bins=30)

axs[0].plot([sd['ConfirmedCases_sd'].mean(), sd['ConfirmedCases_sd'].mean()], [0,30], 'r--', 3, label='Mean')

axs[0].set_title(f"COVID-19 Confirmed Cases on {START_DATE.strftime('%Y-%m-%d')}", size=18)

axs[0].legend() # Mean is mentioned twice on the legend

axs[0].grid()

axs[1].hist(sd['ConfirmedCases_ct'], bins=30)

axs[1].plot([sd['ConfirmedCases_ct'].mean(), sd['ConfirmedCases_ct'].mean()], [0,30], 'r--', 3, label='Mean')

axs[1].set_title(f"COVID-19 Confirmed Cases on {china.Date.max().strftime('%Y-%m-%d')}", size=18)

axs[1].legend() # Mean is mentioned twice on the legend

axs[1].grid()

axs[2].hist(sd['PercentGrowth'], bins=10)

axs[2].plot([sd['PercentGrowth'].mean(), sd['PercentGrowth'].mean()], [0,11], 'r--', 3, label='Mean')

axs[2].set_title('COVID-19 Groth Rate Distribution for Chinese Provenses', size=18)

axs[2].legend() # Mean is mentioned twice on the legend

axs[2].grid()

plt.show()
sd[sd['PercentGrowth'] > 40]
sd[sd['ConfirmedCases_sd'] > 50]
us = train[train['Country/Region'] == 'US']

max_cases = us.groupby('area_full').max()

over_50 = max_cases[max_cases['ConfirmedCases'] > 100].index

# us.groupby('Date').sum()
compare_curves('US_New York', 'Iran_No Sub', )
train.area_full.unique()