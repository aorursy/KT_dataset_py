#Data Source(s): 

#https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

#https://github.com/CSSEGISandData/COVID-19

#https://www.kaggle.com/kimjihoo/coronavirusdataset
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

import os.path

import seaborn as sns

sns.set_style('darkgrid')
COVID19 = "../input/novel-corona-virus-2019-dataset"

SK_COVID19 = "../input/coronavirusdataset"



covid = pd.read_csv(os.path.join(COVID19, 'covid_19_data.csv'), sep=',')

#open_line = pd.read_csv(os.path.join(COVID19, 'COVID19_open_line_list.csv'), sep=',')

#line_list = pd.read_csv(os.path.join(COVID19, 'COVID19_line_list.csv'), sep=',')



global_confirmed = pd.read_csv(os.path.join(COVID19, 'time_series_covid_19_confirmed.csv'), sep=',')

global_recovered = pd.read_csv(os.path.join(COVID19, 'time_series_covid_19_recovered.csv'), sep=',')

global_deaths = pd.read_csv(os.path.join(COVID19, 'time_series_covid_19_deaths.csv'), sep=',')



US_confirmed = pd.read_csv(os.path.join(COVID19, 'time_series_covid_19_confirmed_US.csv'), sep=',')

US_deaths = pd.read_csv(os.path.join(COVID19, 'time_series_covid_19_deaths_US.csv'), sep=',')



SouthKoreadata = pd.read_csv(os.path.join(SK_COVID19, 'Time.csv'), sep=',')
covid
global_confirmed
global_recovered
global_deaths
US_confirmed
US_deaths
SouthKoreadata
#drop all null values

US_confirmed.isnull().any()

US_deaths.isnull().any()

global_confirmed.isnull().any()

global_deaths.isnull().any()

global_recovered.isnull().any()

SouthKoreadata.isnull().any
#US confirmed cases by state (May 2020)

mayUS_confirmed = US_confirmed[['Province_State','5/8/20']]

mayUS_confirmed.groupby(['Province_State']).mean()
USC_state_mean = mayUS_confirmed.groupby(['Province_State']).mean()
#US deaths by state (May 2020)

mayUS_deaths = US_deaths[['Province_State','5/8/20']]

mayUS_deaths.groupby(['Province_State']).mean()
USD_state_mean = mayUS_deaths.groupby(['Province_State']).mean()
#global confirmed cases (May 2020)

mayGC = global_confirmed[['Country/Region','5/8/20']]



#delete US data

mayGC = mayGC.drop([225])



#aggregate by mean (dataframe to series) 

mayGC.groupby(['Country/Region']).mean()
GC_mean = mayGC.groupby(['Country/Region']).mean()
#global death cases (May 2020)

mayGD = global_deaths[['Country/Region','5/8/20']]



#delete US data

mayGD = mayGD.drop([225])



#aggregate by mean (dataframe to series) 

mayGD.groupby(['Country/Region']).mean()
GD_mean = mayGD.groupby(['Country/Region']).mean()
#global recovered cases (May 2020)

mayGR = global_recovered[['Country/Region','5/8/20']]



#delete US data

mayGR = mayGR.drop([225])



#aggregate by mean (dataframe to series) 

mayGR.groupby(['Country/Region']).mean()
GR_mean = mayGR.groupby(['Country/Region']).mean()
#clean up South Korea data

SouthKoreadata[['date','confirmed','released','deceased']]
SouthKoreadata = SouthKoreadata[['date','confirmed','released','deceased']]
#clean up unwanted US confirmed cases data columns

US_confirmed.drop(['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Lat','Long_',

                   'Combined_Key'],axis=1)
US_confirmed = US_confirmed.drop(['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Lat','Long_',

                                  'Combined_Key'],axis=1)
#clean up unwanted US death cases data columns

US_deaths.drop(['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Lat','Long_',

                'Combined_Key','Population'],axis=1)
US_deaths = US_deaths.drop(['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Lat','Long_',

                            'Combined_Key','Population'],axis=1)
#overall US confirmed cases

US_confirmed.groupby(['Country_Region']).sum()
USC_overall = US_confirmed.groupby(['Country_Region']).sum()
#overall US death cases

US_deaths.groupby(['Country_Region']).sum()
USD_overall = US_deaths.groupby(['Country_Region']).sum()
#overall global confirmed/deaths/recovered cases

global_confirmed = global_confirmed.drop(['Province/State','Lat','Long'],axis=1)

global_deaths = global_deaths.drop(['Province/State','Lat','Long'],axis=1)

global_recovered = global_recovered.drop(['Province/State','Lat','Long'],axis=1)
#overall China confirmed cases

China_confirmed = global_confirmed[global_confirmed['Country/Region'].str.contains("China")]

China_confirmed.groupby(['Country/Region']).sum()
ChinaC_overall = China_confirmed.groupby(['Country/Region']).sum()
#overall China death cases

China_deaths = global_deaths[global_deaths['Country/Region'].str.contains("China")]

China_deaths.groupby(['Country/Region']).sum()
ChinaD_overall = China_deaths.groupby(['Country/Region']).sum()
#overall China recovered cases 

China_recovered = global_recovered[global_recovered['Country/Region'].str.contains("China")]

China_recovered.groupby(['Country/Region']).sum()
ChinaR_overall = China_recovered.groupby(['Country/Region']).sum()
#overall Germany confirmed cases

Germany_confirmed = global_confirmed[global_confirmed['Country/Region'].str.contains("Germany")]

Germany_confirmed.mean()
GermanyC_overall = Germany_confirmed.mean()
#overall Germany death cases

Germany_deaths = global_deaths[global_deaths['Country/Region'].str.contains("Germany")]

Germany_deaths.mean()
GermanyD_overall = Germany_deaths.mean()
#overall Germany recovered cases

Germany_recovered = global_recovered[global_recovered['Country/Region'].str.contains("Germany")]

Germany_recovered

Germany_recovered.mean()
GermanyR_overall = Germany_recovered.mean()
#overall South Korea confirmed cases

SouthKoreadata[['date', 'confirmed']]
SouthKorea_confirmed = SouthKoreadata[['date', 'confirmed']]
#overall South Korea deaths

SouthKoreadata[['date', 'deceased']]
SouthKorea_deaths = SouthKoreadata[['date', 'deceased']]
#overall South Korea recovered cases

SouthKoreadata[['date', 'released']]
SouthKorea_recovered = SouthKoreadata[['date', 'released']]
#mortality rate = number of deaths/total population

#population data taken from worldometers.info



#global population = 7,648,716,300

#US total population = 329,628,455

#China total population = 1,349,015,977

#Germany total population = 83,746,284

#South Korea total population = 51,269,185
#mortality rates (by country)

US_mortality = (USD_overall / 764871630)

China_mortality = (ChinaD_overall / 764871630) 

Germany_mortality = (GermanyD_overall / 764871630) 

SouthKorea_deaths['deceased'] = SouthKorea_deaths['deceased'].div(764871630)

SouthKorea_mortality = SouthKorea_deaths
#transpose each one of the mortality data frames
US_mortality.transpose()
China_mortality.transpose()
Germany_mortality
SouthKorea_mortality
US_mortality = US_mortality.transpose()

China_mortality = China_mortality.transpose()
#Data Visualizations

#1) US confirmed (May) VS US deaths (May)

#2) global confirmed (May) VS global deaths (May) VS global recovered (May) -> bar

#3) US overall confirmed vs US deaths overall (over time) -> line

#4) China overall confirmed vs China overall death vs China overall confirmed (over time) -> line

#5) Germany overall confirmed vs Germany overall death vs Germany overall confirmed (over time) -> line

#6) South Korea overall confirmed vs South Korea overall death vs South Korea overall confirmed (over time) -> line

#7) US mortality rate vs China mortality rate vs Germany mortality vs South Korea mortality rate (over time) -> line

#8) China mortality rate vs Germany mortality vs South Korea mortality rate
#Correlation between US confirmed cases and US deaths

np.corrcoef(USC_state_mean['5/8/20'],USD_state_mean['5/8/20'])
#Correlation between global confirmed cases and global deaths

np.corrcoef(GC_mean['5/8/20'],GD_mean['5/8/20'])
#Correlation between global deaths and global recoveries in May

np.corrcoef(GD_mean['5/8/20'],GR_mean['5/8/20'] )
#transpose the overall US confirmed cases over time and US death cases over time data frames

USC_overall = USC_overall.transpose()

USD_overall = USD_overall.transpose()
#Correlation between overall US confirmed cases and US deaths

np.corrcoef(USC_overall['US'],USD_overall['US'])
ChinaC_overall = ChinaC_overall.transpose()

ChinaD_overall = ChinaD_overall.transpose()

ChinaR_overall = ChinaR_overall.transpose()
#Correlation between overall China confirmed cases and China deaths

np.corrcoef(ChinaC_overall['China'],ChinaD_overall['China'])
#Correlation between China deaths and China recoveries

np.corrcoef(ChinaD_overall['China'],ChinaR_overall['China'])
#Correlation between overall Germany confirmed cases and Germany deaths

np.corrcoef(GermanyC_overall, GermanyD_overall)
#Correlation between Germany deaths and Germany recoveries

np.corrcoef(GermanyD_overall, GermanyR_overall)
#Correlation between overall South Korea confirmed cases and South Korea deaths

np.corrcoef(SouthKorea_confirmed['confirmed'], SouthKorea_deaths['deceased'])
#Correlation between overall South Korea deaths and South Korea recoveries

np.corrcoef(SouthKorea_deaths['deceased'], SouthKorea_recovered['released'])
#Data Visualization #1

fig1 = plt.figure(figsize=(18,6))

ax1 = fig1.add_subplot(111)



USC_state_mean.sort_values(by = ['5/8/20']).plot(ax = ax1, kind = 'bar', fontsize = 12, color = 'orange')

USD_state_mean.sort_values(by = ['5/8/20']).plot(ax = ax1, kind = 'bar', fontsize = 12, color = 'purple')



ax1.legend(["Confirmed Cases", "Deaths"])

plt.ylabel('Population', fontsize = 14)

plt.xlabel('US States/Provinces', fontsize = 14)

plt.title('US Confirmed COVID-19 Cases Per State', fontsize = 20)



rcParams["font.family"] = 'monospace'
#Data Visualization #2

fig2 = plt.figure(figsize=(80,40))

ax2 = fig2.add_subplot(111)



GC_mean.sort_values(by = ['5/8/20']).plot(ax = ax2, kind = 'bar', fontsize = 30, color = 'royalblue')

GR_mean.sort_values(by = ['5/8/20']).plot(ax = ax2, kind = 'bar', fontsize = 30, color = 'gold')

GD_mean.sort_values(by = ['5/8/20']).plot(ax = ax2, kind = 'bar', fontsize = 30, color = 'darkred')



ax2.legend(["Confirmed Cases", "Recoveries", "Deaths"], loc=2, prop={'size': 30})

plt.ylabel('Population', fontsize = 40)

plt.xlabel('Countries', fontsize = 40)

plt.title('Global Confirmed COVID-19 Cases vs Global Deaths vs Global Recoveries (sans US)', fontsize = 60)



rcParams["font.family"] = 'monospace'
#Data Visualization #3

fig3 = plt.figure(figsize=(18,6))

ax3 = fig3.add_subplot(111)



USC_overall.sort_values(by = ['US']).plot(ax = ax3, fontsize = 12, color = 'orange', linewidth = 2)

USD_overall.sort_values(by = ['US']).plot(ax = ax3, fontsize = 12, color = 'purple', linewidth = 3)



ax3.legend(["Confirmed Cases", "Deaths"])

plt.ylabel('Population', fontsize = 14)

plt.xlabel('Dates (MM/DD/YY)', fontsize = 14)

plt.title('US Confirmed COVID-19 Cases VS Deaths', fontsize = 20)



rcParams["font.family"] = 'monospace'

#Data Visualization #4

fig4 = plt.figure(figsize=(18,6))



ax4 = fig4.add_subplot(111)



ChinaC_overall.sort_values(by = ['China']).plot(ax = ax4, fontsize = 12, color = 'royalblue', linewidth = 3)

ChinaR_overall.sort_values(by = ['China']).plot(ax = ax4, fontsize = 12, color = 'gold', linewidth = 3)

ChinaD_overall.sort_values(by = ['China']).plot(ax = ax4, fontsize = 12, color = 'darkred', linewidth = 3)



ax4.legend(["Confirmed Cases", "Recoveries", "Deaths"])

plt.ylabel('Population', fontsize = 14)

plt.xlabel('Dates (MM/DD/YY)', fontsize = 14)

plt.title('COVID 19: China Confirmed Cases VS Recovered Cases VS Deaths', fontsize = 20)



rcParams["font.family"] = 'monospace'
#Data Visualization #5

fig5 = plt.figure(figsize=(18,6))



ax5 = fig5.add_subplot(111)



GermanyC_overall.plot(ax = ax5, fontsize = 12, color = 'royalblue', linewidth = 3)

GermanyR_overall.plot(ax = ax5, fontsize = 12, color = 'gold', linewidth = 3)

GermanyD_overall.plot(ax = ax5, fontsize = 12, color = 'darkred', linewidth = 3)



ax5.legend(["Confirmed Cases", "Recoveries", "Deaths"])

plt.ylabel('Population', fontsize = 14)

plt.xlabel('Dates (MM/DD/YY)', fontsize = 14)

plt.title('COVID 19: Germany Confirmed Cases VS Recovered Cases VS Deaths', fontsize = 20)



rcParams["font.family"] = 'monospace'
#Data Visualization #6

fig6 = plt.figure(figsize=(18,6))



ax6 = fig6.add_subplot(111)



SouthKorea_confirmed.plot(ax = ax6, fontsize = 12, color = 'royalblue', linewidth = 3)

SouthKorea_recovered.plot(ax = ax6, fontsize = 12, color = 'gold', linewidth = 3)

SouthKorea_deaths.plot(ax = ax6, fontsize = 12, color = 'darkred', linewidth = 3)



ax6.legend(["Confirmed Cases", "Recoveries", "Deaths"])

plt.ylabel('Population', fontsize = 14)

plt.xlabel('Dates (MM/DD/YY)', fontsize = 14)

plt.title('COVID 19: South Korea Confirmed Cases VS Recovered Cases VS Deaths', fontsize = 20)



rcParams["font.family"] = 'monospace'
#Data Visualization #7

fig7 = plt.figure(figsize=(18,6))



ax7 = fig7.add_subplot(111)



SouthKorea_mortality.plot(ax = ax7, fontsize = 12, color = 'darkseagreen', linewidth = 2)

Germany_mortality.plot(ax = ax7, fontsize = 12, color = 'darkred', linewidth = 2)

US_mortality.plot(ax = ax7, fontsize = 12, color = 'royalblue', linewidth = 2)

China_mortality.plot(ax = ax7, fontsize = 12, color = 'gold', linewidth = 2)





plt.ylabel('Mortality Rate', fontsize = 14)

plt.xlabel('Dates (MM/DD/YY)', fontsize = 14)

plt.title(' COVID-19: US Mortality Rate VS China Mortality Rate VS Germany Mortality Rate VS South Korea Mortality Rate', fontsize = 20)



ax7.legend(["South Korea", "Germany", "US", "China"])



rcParams["font.family"] = 'monospace'
#Data Visualization #8

fig8 = plt.figure(figsize=(18,6))



ax8 = fig8.add_subplot(111)



Germany_mortality.plot(ax = ax8, fontsize = 12, color = 'darkred', linewidth = 2)

SouthKorea_mortality.plot(ax = ax8, fontsize = 12, color = 'darkseagreen', linewidth = 2)

China_mortality.plot(ax = ax8, fontsize = 12, color = 'gold', linewidth = 2)





plt.ylabel('Mortality Rate', fontsize = 14)

plt.xlabel('Dates (MM/DD/YY)', fontsize = 14)

plt.title(' COVID-19: China Mortality Rate VS South Korea Mortality Rate vs Germany Mortality Rate', fontsize = 20)



ax8.legend([ "Germany", "US", "China"])



rcParams["font.family"] = 'monospace'