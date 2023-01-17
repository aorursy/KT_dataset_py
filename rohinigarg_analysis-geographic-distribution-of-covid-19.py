#Author : Rohini Garg

import numpy as np

import pandas as pd

# 1.2 For plotting

import matplotlib.pyplot as plt

#import matplotlib

#import matplotlib as mpl     # For creating colormaps

import seaborn as sns

# 1.3 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4 OS related

import os



%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

plt.style.use('dark_background')





#os.chdir("/home/puneet/Desktop/Notebook/COVID")



os.chdir("/kaggle/input/uncover/ECDC")

os.listdir()            # List all files in the folder



#load data

df = pd.read_csv("/kaggle/input/uncover/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv", parse_dates=['daterep'])

#df.head()



df[df['continentexp'] == 'Asia']['geoid'].unique()

#df.columns.values



df['year'].unique()    # Data of two years

df['month'].unique()   # Data from december to april

df['daterep'] = pd.to_datetime(df['year'] * 10000 + df['month'] * 100 + df['day'], format='%Y%m%d')

#df['month'] = df['month'].map({ 12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'April' })

import calendar

df['month']=df['month'].apply(lambda x: calendar.month_abbr[x])

df['month']=df['month'] + '-' + df.year.astype(str)

df.sort_values(by = 'daterep', inplace=True, ascending=True)

df['month_no']=df.daterep.dt.month



df.head

df['cum_cases'] = 0

df['cum_deaths'] = 0

df.rename({

    'countriesandterritories': 'country', 

    'popdata2018': 'pop', 

    'countryterritorycode': 'code',

    'continentexp': 'cont'}, axis=1, inplace=True)

#

#  Cummulative cases, deaths for each country

#

gr = df.groupby('country')

for country in df['country'].unique():

    g = gr.get_group(country)

    df.loc[df.country == country, 'cum_cases'] = g['cases'].cumsum(axis=0)

    df.loc[df.country == country, 'cum_deaths'] = g['deaths'].cumsum(axis=0)

df['mort_rate'] = df['cum_deaths'] / df['cum_cases'] * 100

df.loc[ ~np.isfinite(df['mort_rate']), 'mort_rate' ] = np.nan

df.fillna({'mort_rate': 0}, inplace=True)   # Remove Nan

import math



grpCountry = df.groupby(['country', 'month'])



totalMonths = df.month.unique().size

nrows = math.ceil(totalMonths / 2)

fig, ax = plt.subplots(nrows, 2, figsize=(10, 10))

ax = ax.ravel()

fig.suptitle("Total cases in Afganistan")

for idx, month in enumerate(df.month.unique()):

    dfTemp = grpCountry.get_group(('Afghanistan', month))

    ax[idx].plot('day', 'cum_cases', 'go--', data=dfTemp)

    ax[idx].set(title='Total cases till {} is {}'.format(month, dfTemp['cum_cases'].max()), xlabel='#Days', ylabel='# Cases')

    

plt.subplots_adjust(wspace = 0.5, hspace = 0.5)

#Observation: Cases are increasing since Mid Mar 2020
import math

grpCountry=df.groupby(['country','month'])

noofmonths=len(df['month'][df.country=='India'].unique())

noofrows=math.ceil(noofmonths/2)

fg,ax=plt.subplots(noofrows,2,figsize=(10,10))

fg.suptitle("Deaths in India")

ax=ax.ravel()

for idx,month in enumerate (df['month'][df.country=='India'].unique()):

    dfIndiaMonth=grpCountry.get_group(('India',month))

    ax[idx].plot('day','cum_deaths','--go',data=dfIndiaMonth)

    ax[idx].set(title="Month {} Total Deaths {}".format(month,dfIndiaMonth.cum_deaths.max()),xlabel='# Days',ylabel="# Deaths")

plt.subplots_adjust(wspace=.5,hspace=.5)

#Observations : Deaths are increasing day by day since  5th April 2020

    
# Mortality rates in india

dfIndia = df.loc[df.country == 'India', :]

fig = plt.figure(figsize=(17, 5))

plt.plot('daterep', 'mort_rate', 'ro--', data = dfIndia)

plt.xlabel("#Day")

plt.ylabel("Mort Rate%")

plt.title("Mortality rate in India")

plt.grid(True)

#Mortality rate is around 3.2 in the end of april
fig = plt.figure(figsize=(17, 15))



#  Total cases

ax = plt.subplot(3, 1, 1)

ax.plot('daterep', 'cum_cases', 'go--', data=dfIndia)

ax.set(title="Total cases in India", xlabel='Date', ylabel='# cases')

plt.grid(True)



#  Total deaths

ax = plt.subplot(3, 1, 2)

ax.plot('daterep', 'cum_deaths', 'yo--', data=dfIndia)

ax.set(title="Total deaths in India", xlabel='Date', ylabel='# deaths')

plt.grid(True)



#  Mortality rate

ax = plt.subplot(3, 1, 3)

ax.plot('daterep', 'mort_rate', 'ro--', data=dfIndia)

ax.set(title="Mortality rate in India", xlabel='Date', ylabel='Mortality rate')

plt.grid(True)

# Drastic change in curve of cases and deaths since second week of April 2020
import seaborn as sns
#cases everymonth continent wise

fig=plt.figure(figsize=(17,8))

sns.barplot(x = 'cont',

            y = 'cases',

            hue = 'month',      

            estimator = np.sum,

            ci = 65,

            data =df)



# America is most affected cont.
fig=plt.figure(figsize=(20,20))

ax=sns.relplot(x="month", y="mort_rate", hue="cont", kind="line",data=df);

ax.set( ylabel="Mortaility rate%")

#America and #Africa has highest mortality rate in April 2020
fig=plt.figure(figsize=(8,10))

gr=df.groupby(['month'])

Total_deaths_cases_month=gr.agg(Total_Death = ('deaths','sum'),Total_cases = ('cases','sum')).reset_index()

x = np.arange(Total_deaths_cases_month.shape[0])

Tot_cases_list=Total_deaths_cases_month.Total_cases.tolist()

Tot_death_list=Total_deaths_cases_month.Total_Death.tolist()

month_list=Total_deaths_cases_month.month.tolist()

plt.bar(x,Tot_cases_list, width=0.25, label='Total cases')

plt.bar(x+.25,Tot_death_list, width=0.25, label='Total Deaths')

plt.xticks(ticks=x,labels=month_list)

plt.ylabel('No of Cases')

plt.title('Wordwide No of Deaths Vs No of Cases Comparision for each month')

plt.legend()

plt.grid(True)

# Cases increased exponentially in the month of April 2020
#all country cases

fig=plt.figure(figsize=(17,17))

gr_country_month=df.groupby(['country','month'])

gr_country_month_agg=gr_country_month.cases.sum().unstack()

sns.heatmap(gr_country_month_agg, cmap = plt.cm.coolwarm)

#United State of America is most affected Country with CO-VID 19
import plotly.express as px



import plotly.graph_objects as go 



df['month_no']=df.daterep.dt.month

gr_cont_month=df.groupby(['country','cont','month','month_no','year'])

df_cont_month=gr_cont_month.agg(Total_Deaths = ('deaths','sum'),Total_cases = ('cases','sum')).reset_index()





df_cont_month["mort_rate"]=df_cont_month.Total_Deaths/df_cont_month.Total_cases *100



df_cont_month.loc[ ~np.isfinite(df_cont_month['mort_rate']), 'mort_rate' ] = np.nan

df_cont_month.fillna({'mort_rate': 0}, inplace=True)  

df_cont_month=df_cont_month[df_cont_month.Total_cases > 0]

df_cont_month[df_cont_month.mort_rate < 0]

df_cont_month.sort_values(by=['year','month_no'],ascending=[True,True],inplace=True)

df_cont_month.head()



fig=px.scatter(df_cont_month,

           x="Total_cases",

           y="Total_Deaths",

           size="mort_rate",

           size_max=60,

           color="cont",

           hover_name="country",

           animation_frame="month",

           animation_group="country",

           log_x=True,

           range_x=[100,100000],

           range_y=[25,90]

           )

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.show()