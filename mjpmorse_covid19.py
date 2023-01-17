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
import pandas as pd

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.style.use('ggplot')

import matplotlib.cm as cm

import seaborn as sns



import pandas_profiling

import numpy as np

from numpy import percentile

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,

                               AutoMinorLocator)



#import reverse_geocoder as rg



import os, sys

import calendar



import warnings

warnings.filterwarnings('ignore')



plt.rc('font', size=18)        

plt.rc('axes', titlesize=22)      

plt.rc('axes', labelsize=18)      

plt.rc('xtick', labelsize=12)     

plt.rc('ytick', labelsize=12)     

plt.rc('legend', fontsize=12)   



plt.rcParams['font.sans-serif'] = ['Verdana']



# function that converts to thousands

# optimizes visual consistence if we plot several graphs on top of each other

def format_1000(value, tick_number):

    return int(value / 1_000)



pd.options.mode.chained_assignment = None

pd.options.display.max_seq_items = 500

pd.options.display.max_rows = 500

pd.set_option('display.float_format', lambda x: '%.5f' % x)
covid_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

covid_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
covid_conf_country=covid_confirmed.drop(['Lat','Long'],axis=1).groupby("Country/Region").sum()

covid_deaths_country=covid_deaths.drop(['Lat','Long'],axis=1).groupby("Country/Region").sum()
sns.set()

#plt.tight_layout()

log_set = True

marker_sz = 15



fig, ax = plt.subplots(figsize=(20,10))

plt.plot(covid_conf_country.loc['China'],marker='.',markersize=marker_sz,ls='None')

plt.plot(covid_deaths_country.loc['China'],marker='X',ls='None')



plt.plot(covid_conf_country.loc['US'],marker='.', markersize=marker_sz,ls='None')

plt.plot(covid_deaths_country.loc['US'],marker='X',ls='None')



plt.yscale('log')

plt.xticks(rotation=90) 

plt.legend(('China Confirmed','China Deaths','US Confirmed','US Deaths'),fontsize='large')



## Hid every other tick

for label in ax.xaxis.get_ticklabels()[::2]:

    label.set_visible(False)



plt.title('Covid19 in US and China')    

plt.show()

#plt.savefig('covid19',dpi=300)
mean_probablity_death=(covid_deaths_country.iloc[:,-1]/covid_conf_country.iloc[:,-1]).fillna(0)
mean_probablity_death_plot=mean_probablity_death.loc[['US','China','Italy','Japan','Taiwan*','Iran','United Kingdom','Iceland','Russia']].sort_values()

mean_probablity_death_plot['World Average'] = (covid_deaths_country.iloc[:,-1].sum()/covid_conf_country.iloc[:,-1].sum())

mean_probablity_death_plot.sort_values(inplace=True)

fig, ax = plt.subplots(figsize=(20,10))

plt.plot(mean_probablity_death_plot,marker='.', markersize=marker_sz,ls='None')

plt.xticks(rotation=90) 

plt.yscale('linear')

plt.title('P(Death|Test Positive) as of 3/26/2020')

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

#plt.show()

plt.savefig('covid19_Prob',dpi=300)
mean_probablity_death_time_avg=(covid_deaths_country/covid_conf_country).fillna(0)
fig, ax = plt.subplots(figsize=(20,10))

#plt.plot(mean_probablity_death_time_avg.loc['US'][mean_probablity_death_time_avg.columns[::2]],marker='.', markersize=marker_sz,ls='None')



plt.plot(mean_probablity_death_time_avg.loc['Italy'][mean_probablity_death_time_avg.columns[::2]],marker='.', markersize=marker_sz,ls='None')



#plt.plot(mean_probablity_death_time_avg.loc['Taiwan*'][mean_probablity_death_time_avg.columns[::2]],marker='.', markersize=marker_sz,ls='None')



#plt.plot(mean_probablity_death_time_avg.loc['China'][mean_probablity_death_time_avg.columns[::2]],marker='.', markersize=marker_sz,ls='None')



plt.xticks(rotation=90) 

plt.yscale('linear')

plt.title('P(Death|Test Positive) in Italy')

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

#plt.legend(('US','Italy','Taiwan*','China'),fontsize='large')

plt.legend('Italy')

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])



#plt.show()

plt.savefig('covid19_Italy',dpi=300)
for country in ['US','Italy','Taiwan*','China','Korea, South']:

    fig, ax = plt.subplots(figsize=(20,10))

    plt.plot(mean_probablity_death_time_avg.loc[country][mean_probablity_death_time_avg.columns[::2]],marker='.', markersize=marker_sz,ls='None')

    plt.xticks(rotation=90) 

    plt.yscale('linear')

    plt.title('P(Death|Test Positive) in %s' %(country))

    vals = ax.get_yticks()

    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    #plt.legend(('US','Italy','Taiwan*','China'),fontsize='large')

    #plt.legend('Italy')

    vals = ax.get_yticks()

    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    plt.savefig('covid19_time_series_%s' %(country),dpi=300)
