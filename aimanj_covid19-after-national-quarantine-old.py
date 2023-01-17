# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv',parse_dates=['Date']);

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv',parse_dates=['Date']);



quarantinesOutsideChina = pd.read_csv('../input/covid19-quarantine-outside-china/covid19_quarantine_outside_China.csv',parse_dates=['Start date', 'End date']);
train.head()
# Sum countries with states, not dealing with states for now

df = train[['Country_Region','Date','ConfirmedCases','Fatalities']].groupby(['Country_Region','Date'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum'})



# change to datetime format

df['Date'] = pd.to_datetime(df['Date'])
# df.sort_values(['Country/Region','Date']).groupby('Country/Region')['Date'].diff().dt.days
print("Start at", min(df['Date']), "to day", max(df['Date']), ", a total of", df['Date'].nunique(), "days")
df.rename(columns={ 

                     'Province_State':'State',

                     'Country_Region':'Country'

                    }, inplace=True)
def p2f(x):

    """

    Convert urban percentage to float

    """

    try:

        return float(x.strip('%'))/100

    except:

        return np.nan



def age2int(x):

    """

    Convert Age to integer

    """

    try:

        return int(x)

    except:

        return np.nan



def fert2float(x):

    """

    Convert Fertility Rate to float

    """

    try:

        return float(x)

    except:

        return np.nan





countries_df = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv", converters={'Urban Pop %':p2f,

                                                                                                             'Fert. Rate':fert2float,

                                                                                                             'Med. Age':age2int,

                                                                                                            'Density (P/Km²)': fert2float})

countries_df.rename(columns={'Country (or dependency)': 'Country',

                             'Population (2020)' : 'Population',

                             'Density (P/Km²)' : 'Density',

                             'Fert. Rate' : 'Fertility',

                             'Med. Age' : "Age",

                             'Urban Pop %' : 'Urban_percentage'}, inplace=True)







countries_df['Country'] = countries_df['Country'].replace('United States', 'US')

countries_df['Country'] = countries_df['Country'].replace('South Korea', 'Korea, South')

countries_df = countries_df[["Country", "Population", "Density", "Fertility", "Age", "Urban_percentage"]]



countries_df.head()
data = pd.merge(df, countries_df, on='Country')
# Sum all confirmed case, fatalities and list them in descending order

country_data = data.fillna('').groupby(['Country'])['ConfirmedCases', 'Fatalities'].max().sort_values(by='ConfirmedCases', ascending=False)

country_data.head(10)
# New data, ConfirmPerDensity

data['ConfirmPerDensity'] = data['ConfirmedCases']/data['Density']

data['ConfirmedPerMilPop'] = data['ConfirmedCases']/data['Population']*1e6

# data['Log_ConfirmPerMilPop'] = np.log(data['ConfirmedPerMilPop'])
# Specify interested countries to analyse and no of cases since outbreak to analyse

listOfCountries = ['Malaysia','Portugal','New Zealand','Australia','India','United Kingdom','Belgium']

NoOfCases = 100



# List_1 = ['Malaysia','Denmark','Italy','Spain','France','New Zealand']

interestCountries = '|'.join(listOfCountries)



# Now get the interested countries from my raw data

selectedCountries = data[data['Country'].str.contains(interestCountries)]



# Normalized so I can see since the first day

selectedCountries = selectedCountries[(selectedCountries[['ConfirmedCases']] >= NoOfCases).all(axis=1)]



# Get the time delta between each days for each country (supposed to be 1 for all)

selectedCountries['TimeDelta'] = selectedCountries.sort_values(['Country','Date']).groupby('Country')['Date'].diff().dt.days



# Get the daily confirm cases

selectedCountries['DailyConfirmedCases'] = selectedCountries.sort_values(['Country','Date']).groupby('Country')['ConfirmedCases'].diff()



# Calculate the cumulative time delta

selectedCountries["cum_sum"] = selectedCountries["TimeDelta"].groupby(selectedCountries['Country']).cumsum()



##### Want ConfirmedCases in selectedCountries for Country and Start Date in selectedCountries ####### 

# Make a loop to iterate over the interested countries

a=[] # Initiating a variable to save data later

for interestCountry in listOfCountries:

    try:

        # Using the country to get the start date of quarantine

        DateAtQ = quarantinesOutsideChina.loc[quarantinesOutsideChina['Country'] == interestCountry, 'Start date'].iloc[0]

        # Find out the Covid data on interested country when quarantine was started

        CovidDataofCountryAtQDay = selectedCountries.loc[(selectedCountries['Country']==interestCountry) & (selectedCountries['Date']==DateAtQ)]

        cum_sumAtQ = CovidDataofCountryAtQDay['cum_sum'].iloc[0]

        ConfirmedCasesAtQ = CovidDataofCountryAtQDay['ConfirmedCases'].iloc[0]

        ConfirmedPerMilPopAtQ = CovidDataofCountryAtQDay['ConfirmedPerMilPop'].iloc[0]

        DailyConfirmedCasesAtQ = CovidDataofCountryAtQDay['DailyConfirmedCases'].iloc[0]

        # Save the data in a new matrix which will be converted to dataframe later

        a.append([interestCountry, DateAtQ, ConfirmedCasesAtQ, cum_sumAtQ, ConfirmedPerMilPopAtQ, DailyConfirmedCasesAtQ])

    except:

        # If theres no lockdown/quarantine for that country, then print this:

        print('There is no quarantine for:', interestCountry)



# Saving the data        

quarantine = pd.DataFrame(a,columns=['Country', 'Date', 'ConfirmedCases','cum_sum','ConfirmedPerMilPop','DailyConfirmedCases'])

quarantine['Date'] = pd.to_datetime(quarantine['Date'])

# quarantine

#Plotting the confirm/density against Date with hue on country

import datetime

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pyplot, dates

# Scaler for log-space on y-axis

from matplotlib.ticker import ScalarFormatter



# format dates for lmplot

selectedCountries['Datenum'] = dates.date2num(selectedCountries['Date'])

# Make a function to change the datenum to date later

@pyplot.FuncFormatter

def fake_dates(x, pos):

    """ Custom formater to turn floats into e.g., 2016-05-08"""

    return dates.num2date(x).strftime('%Y-%m-%d')



y_axis = 'DailyConfirmedCases'

# Set pallete colour for each country at random

unique = selectedCountries["Country"].unique()

palette = dict(zip(unique, sns.color_palette()))

palette.update({"Malaysia":"k"})



# Plotting the data using a lineplot

fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111)

sns.set(font_scale=2)

sns.set_style("white")

sns.scatterplot('Date', 'DailyConfirmedCases', hue='Country',style="Country",markers=True, data=selectedCountries,lw=3,s=100, palette=palette)

ax.set_xlim(datetime.date(2020, 3, 16), datetime.date(2020, 4, 10))

## Transofrm the y-axis scale to logspacing scale as this gives a better visual representation

## dir(ax.ax.set_yscale("log"))

## dir(ax.ax.get_yaxis().set_major_formatter(ScalarFormatter()))

## plt.ticklabel_format(style = 'plain')

ax.set(ylim = (1,1000))



# ax = sns.lmplot('Datenum', 'DailyConfirmedCases', hue='Country',

#                 data=selectedCountries, ci=None, order=5, truncate=True,

#                 palette=palette, legend=False,size=5, aspect=2)

# dir(ax.ax.get_xaxis().set_major_formatter(fake_dates))

# dir(ax.ax.set_xlim(datetime.date(2020, 3, 18), datetime.date(2020, 4, 15)))

# dir(ax.ax.set(ylim = (1,1000)))









# Now plot the point of national quarantine 

for interestCountry,colorLine in palette.items():

    try:

        pointOfXPlot = quarantine.loc[quarantine['Country']==interestCountry,'Date'].iloc[0]

        ax.axvline(pointOfXPlot, color=colorLine, linestyle="-",lw=5,alpha=0.5)

    except:

        print('no quarantine data for',interestCountry)    





# Add labels to the graph and also the legend

# plt.xlabel('Days since >%d case(s) found in the country' %NoOfCases)

plt.xticks(rotation=45)

plt.ylabel('Daily confirm cases')

leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),markerscale=2)

sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)



# set the linewidth of each legend object

# for legobj in leg.legendHandles:

#     legobj._legmarker.set_markersize(9)

#     legobj.set_linewidth(5.0)



# Remove border as it looks betteer without it



plt.show()
#Plotting the confirm/density against Date with hue on country

import datetime

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pyplot, dates

# Scaler for log-space on y-axis

from matplotlib.ticker import ScalarFormatter



# format dates for lmplot

selectedCountries['Datenum'] = dates.date2num(selectedCountries['Date'])

# Make a function to change the datenum to date later

@pyplot.FuncFormatter

def fake_dates(x, pos):

    """ Custom formater to turn floats into e.g., 2016-05-08"""

    return dates.num2date(x).strftime('%Y-%m-%d')



y_axis = 'DailyConfirmedCases'

# Set pallete colour for each country at random

unique = selectedCountries["Country"].unique()

palette = dict(zip(unique, sns.color_palette()))

palette.update({"Malaysia":"k"})



# Plotting the data using a lineplot

fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111)

sns.set(font_scale=2)

sns.set_style("white")

sns.scatterplot('Date', 'DailyConfirmedCases', hue='Country',style="Country",markers=True, data=selectedCountries,lw=3,s=100, palette=palette)

ax.set_xlim(datetime.date(2020, 3, 16), datetime.date(2020, 4, 10))

## Transofrm the y-axis scale to logspacing scale as this gives a better visual representation

ax.set_yscale("log")

ax.get_yaxis().set_major_formatter(ScalarFormatter())

## plt.ticklabel_format(style = 'plain')

ax.set(ylim = (1,10000))



# Now plot the point of national quarantine 

for interestCountry,colorLine in palette.items():

    try:

        pointOfXPlot = quarantine.loc[quarantine['Country']==interestCountry,'Date'].iloc[0]

        ax.axvline(pointOfXPlot, color=colorLine, linestyle="-",lw=5,alpha=0.5)

    except:

        print('no quarantine data for',interestCountry)    





# Add labels to the graph and also the legend

# plt.xlabel('Days since >%d case(s) found in the country' %NoOfCases)

plt.xticks(rotation=45)

plt.ylabel('Daily confirm cases (log scale)')

leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),markerscale=2)

sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)



# set the linewidth of each legend object

# for legobj in leg.legendHandles:

#     legobj._legmarker.set_markersize(9)

#     legobj.set_linewidth(5.0)



# Remove border as it looks betteer without it



plt.show()