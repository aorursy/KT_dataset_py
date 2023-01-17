# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install chart_studio

!pip install plotly-geo
import chart_studio.plotly as py

import plotly.tools as tls

import plotly.graph_objs as go

import plotly

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

plotly.offline.init_notebook_mode(connected=True)
majordir='/kaggle/input/covid19-us-county-trend/'

datadir=majordir+'csse_covid_19_daily_reports/'

date_today=28
covid_data_world_daily_0322=pd.read_csv(datadir+'03-22-2020.csv')

covid_data_world_daily_0322.rename(columns={'Confirmed':'Confirmed_0322'},inplace=True)

covid_data_world_daily_0322.rename(columns={'Deaths':'Deaths_0322'},inplace=True)

covid_data_us_daily_0322=covid_data_world_daily_0322[covid_data_world_daily_0322['Country_Region']=='US'].copy()

covid_data_us_daily_0322.shape
covid_data_us_daily_0322= covid_data_us_daily_0322[covid_data_us_daily_0322['FIPS'].notna()]

vc=covid_data_us_daily_0322['FIPS'].value_counts()

vclist=vc[vc > 1].index.tolist()

covid_data_us_daily_0322=covid_data_us_daily_0322[~(covid_data_us_daily_0322['FIPS'].isin(vclist)&(covid_data_us_daily_0322['Confirmed_0322']>0))]

covid_data_us_daily_0322.shape
for i in range(23,date_today):

    dataset=datadir+'03-'+str(i)+'-2020.csv'

    colc='Confirmed_03'+str(i)

    covid_data_world_daily=pd.read_csv(dataset)

    covid_data_world_daily.rename(columns={'Confirmed':colc},inplace=True)

    

    cold='Deaths_03'+str(i)

    covid_data_world_daily.rename(columns={'Deaths':cold},inplace=True)

    

    covid_data_us_daily=covid_data_world_daily[covid_data_world_daily['Country_Region']=='US'].copy()    



    if i==23:

        print(i)

        covid_data_us_dailytrend=covid_data_us_daily_0322[['FIPS','Confirmed_0322','Deaths_0322']].merge(covid_data_us_daily[['FIPS',colc,cold]],on='FIPS').dropna()

    else:

        print(i)

        covid_data_us_dailytrend=covid_data_us_dailytrend.merge(covid_data_us_daily[['FIPS',colc,cold]],on='FIPS').dropna()

covid_data_us_dailytrend.shape
covid_data_us_dailytrend= covid_data_us_dailytrend[covid_data_us_dailytrend['FIPS'].notna()]

covid_data_us_dailytrend.shape
covid_data_us_dailytrend=covid_data_us_dailytrend.drop_duplicates(['FIPS'])

vc=covid_data_us_dailytrend['FIPS'].value_counts()

vclist=vc[vc > 1].index.tolist()

vc[vc > 1]
covid_data_us_dailytrend[covid_data_us_dailytrend['FIPS'].isin(vclist)]
census_df_fips = pd.read_excel(majordir+'PopulationEstimates_us_county_level_2018.xlsx',skiprows=1)

census_df_fips.FIPS=census_df_fips.FIPS.astype(float)

census_density_df_fips = pd.read_csv(majordir+'uscounty_populationdesity.csv', encoding = "ISO-8859-1",skiprows=1)

census_density_df_fips.rename(columns={'Target Geo Id2':'FIPS'},inplace=True)

census_pop_density_df_fips=census_df_fips.merge(census_density_df_fips[['FIPS','Density per square mile of land area - Population']],on='FIPS')

census_pop_density_df_fips.shape
census_pop_density_df_fips.head()
df_icubeds = pd.read_csv('../input/fork-from-icu-beds-per-county-in-the-us-map/ICU_beds.csv')

print(df_icubeds.shape)

df_icubeds.rename(columns={'fips':'FIPS'},inplace=True)

census_pop_density_df_fips_icu_beds=census_pop_density_df_fips.merge(df_icubeds[['ICU Beds','Population Aged 60+','Percent of Population Aged 60+','Residents Aged 60+ Per Each ICU Bed','FIPS']],on='FIPS')

print(census_pop_density_df_fips_icu_beds.shape)
census_pop_density_df_fips_covid_icu_beds=census_pop_density_df_fips_icu_beds.merge(covid_data_us_dailytrend,on='FIPS')

census_pop_density_df_fips_covid_icu_beds.head()
for i in range(22,date_today):

    colc='Confirmed_03'+str(i)

    colc_10000='Confirmed_per10000_03'+str(i)

    cold='Deaths_03'+str(i)

    cold_100000='Deaths_per100000_03'+str(i)

    census_pop_density_df_fips_covid_icu_beds[colc_10000]=10000*(census_pop_density_df_fips_covid_icu_beds[colc]/census_pop_density_df_fips_covid_icu_beds['POP_ESTIMATE_2018'])

    census_pop_density_df_fips_covid_icu_beds[cold_100000]=100000*(census_pop_density_df_fips_covid_icu_beds[cold]/census_pop_density_df_fips_covid_icu_beds['POP_ESTIMATE_2018'])

   
fips = census_pop_density_df_fips_covid_icu_beds.FIPS.tolist()

values =census_pop_density_df_fips_covid_icu_beds.Confirmed_per10000_0322.tolist()



fig = ff.create_choropleth(fips=fips, values=values)

fig.layout.template = None

fig.show()
print(date_today-1)

col='Confirmed_per10000_03'+str(date_today-1)

fips = census_pop_density_df_fips_covid_icu_beds.FIPS.tolist()

values =census_pop_density_df_fips_covid_icu_beds[col].tolist()



fig = ff.create_choropleth(fips=fips, values=values)

fig.layout.template = None

fig.show()
fips = census_pop_density_df_fips_covid_icu_beds.FIPS.tolist()

values =census_pop_density_df_fips_covid_icu_beds.Deaths_per100000_0322.tolist()



fig = ff.create_choropleth(fips=fips, values=values)

fig.layout.template = None

fig.show()
col='Deaths_per100000_03'+str(date_today-1)

fips = census_pop_density_df_fips_covid_icu_beds.FIPS.tolist()

values =census_pop_density_df_fips_covid_icu_beds[col].tolist()



fig = ff.create_choropleth(fips=fips, values=values)

fig.layout.template = None

fig.show()
col_Deaths_today='Deaths_03'+str(date_today-1)

Deaths_today_1=census_pop_density_df_fips_covid_icu_beds[census_pop_density_df_fips_covid_icu_beds[col_Deaths_today]>0].dropna().copy()



col_Deaths_per100000_today='Deaths_per100000_03'+str(date_today-1)

X = Deaths_today_1['Residents Aged 60+ Per Each ICU Bed'].values.reshape(-1, 1)  # values converts it into a numpy array

Y = Deaths_today_1[col_Deaths_per100000_today].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

linear_regressor = LinearRegression()  # create object for the class

linear_regressor.fit(X, Y)  # perform linear regression

Y_pred = linear_regressor.predict(X)  # make predictions



plt.plot(Deaths_today_1['Residents Aged 60+ Per Each ICU Bed'],Deaths_today_1[col],'*')

plt.xscale('log')

plt.yscale('log')

plt.xlabel('Residents Aged 60+ Per Each ICU Bed')

plt.ylabel(col_Deaths_per100000_today)

plt.plot(X, Y_pred,'*')
linear_regressor.intercept_,linear_regressor.coef_,mean_squared_error(Y,Y_pred),r2_score(Y,Y_pred)
Deaths_today_1.sort_values(by=col_Deaths_today,ascending =False).head().T
import ipywidgets as widgets

from ipywidgets import interact, interact_manual, Layout, interactive
census_pop_density_df_fips_covid_icu_beds[census_pop_density_df_fips_covid_icu_beds.FIPS==11001].drop_duplicates(inplace=True)
# state_counts=census_pop_density_df_fips_covid_icu_beds.groupby('State')[['POP_ESTIMATE_2018','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327']].mean()

state_counts=census_pop_density_df_fips_covid_icu_beds.groupby('State')[['POP_ESTIMATE_2018','Confirmed_0322','Confirmed_0323','Confirmed_0324','Confirmed_0325','Confirmed_0326','Confirmed_0327']].sum()

state_counts.columns=['POP_ESTIMATE_2018',22,23,24,25,26,27]

for i in range(22,date_today):

    state_counts[i]=10000*state_counts[i]/state_counts['POP_ESTIMATE_2018']

state_counts.sort_values(by=27,ascending=False).head()

state_counts_per10000pop=state_counts[[22,23,24,25,26,27]].copy()

state_counts_per10000pop.sort_values(by=27,ascending=False).head()



# state_counts_per10000pop.drop('NY',inplace=True)

state_counts_per10000pop.loc['norm'] = state_counts_per10000pop.max().max()

state_counts_per10000pop.tail()

days=[22,23,24,25,26,27]
# # get list of the years in order, easiest if manually created vs. search and using list(set())

# days = [i for i in range(22, date_today)]



# #create data frame with index=states and columns=year

# county_counts = pd.DataFrame(index=census_pop_density_df_fips_covid_icu_beds.FIPS.value_counts().sort_index().index.tolist(), columns=days)

# # fill each year column with the number of contaminated sites in each state

# for i in years: state_counts[i] = data[data.YEAR == str(i)].ST.value_counts()



# create a list and loop through every year, store the trace in data_bal and then update with a 

# new year will have a list with a trace for every year

data_bal = []

for i in days:

    data_upd = [dict(type='choropleth',

                     name=i,

                     reversescale=True,

                     locations = state_counts_per10000pop[i].index,

                     z = state_counts_per10000pop[i].values,

                     locationmode = 'USA-states',

                     colorbar = dict(title='# Confirmed cases per 10000 people')

                    )

               ]

    

    data_bal.extend(data_upd)

    

# set menus inside the plot

# Create list called 'Steps', where each element is a boolean list indicating which trace 

# in data_bal should be used. The length of data_bal = number of years in the slider, so for 

# each year on the slider we will have a boolean list that is the length of 'years', with 

# every value set to 'False', except for the element corresponding to the trace for that year, 

# which we set with 'step['arg'][1][i]=True'. Each list will be called with the slider to

# tell plotly which trace with show for that slider option. The 'restyle' method means we are

# editting data in the plot, and the 'visible' argument is the bool array mentioned previously.

steps = []

for i in range(0,len(data_bal)):

    step = dict(method = "restyle",

                args = ["visible", [False]*len(data_bal)],

                label = days[i]) 

    step['args'][1][i] = True

    steps.append(step)



# Sliders layout:

sliders = [dict(active = 10,

                currentvalue = {"prefix": "Day: "},

                pad = {"t": 50},

                steps = steps)]



# Plot layout

layout = dict( geo = dict(scope='usa',

                         projection=dict( type='albers usa')),

              sliders = sliders)



fig = dict(data=data_bal, layout=layout)

iplot(fig)
# state_counts=census_pop_density_df_fips_covid_icu_beds.groupby('State')[['POP_ESTIMATE_2018','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327']].mean()

state_counts=census_pop_density_df_fips_covid_icu_beds.groupby('State')[['POP_ESTIMATE_2018','Confirmed_0322','Confirmed_0323','Confirmed_0324','Confirmed_0325','Confirmed_0326','Confirmed_0327']].sum()

state_counts.columns=['POP_ESTIMATE_2018',22,23,24,25,26,27]

for i in range(22,date_today):

    state_counts[i]=10000*state_counts[i]/state_counts['POP_ESTIMATE_2018']

state_counts.sort_values(by=27,ascending=False).head()

state_counts_per10000pop=state_counts[[22,23,24,25,26,27]].copy()

state_counts_per10000pop.sort_values(by=27,ascending=False).head()



state_counts_per10000pop.drop('NY',inplace=True)

state_counts_per10000pop.loc['norm'] = state_counts_per10000pop.max().max()

state_counts_per10000pop.tail()
data_bal = []

for i in days:

    data_upd = [dict(type='choropleth',

                     name=i,

                     reversescale=True,

                     locations = state_counts_per10000pop[i].index,

                     z = state_counts_per10000pop[i].values,

                     locationmode = 'USA-states',

                     colorbar = dict(title='# Confirmed cases per 10000 people')

                    )

               ]

    

    data_bal.extend(data_upd)



steps = []

for i in range(0,len(data_bal)):

    step = dict(method = "restyle",

                args = ["visible", [False]*len(data_bal)],

                label = days[i]) 

    step['args'][1][i] = True

    steps.append(step)



# Sliders layout:

sliders = [dict(active = 10,

                currentvalue = {"prefix": "Day: "},

                pad = {"t": 50},

                steps = steps)]



# Plot layout

layout = dict( geo = dict(scope='usa',

                         projection=dict( type='albers usa')),

              sliders = sliders)



fig = dict(data=data_bal, layout=layout)

iplot(fig)