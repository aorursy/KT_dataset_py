# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

#           Uncomment the line below to view Filenames

#           print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# DATA_PATH = "../input/CORD-19-research-challenge/"

CLEAN_DATA_PATH = "../input/cord-19-eda-parse-json-and-generate-clean-csv/"





# Read in the papers from each source to a dataframe

pmc_df = pd.read_csv(CLEAN_DATA_PATH + "clean_pmc.csv")

biorxiv_df = pd.read_csv(CLEAN_DATA_PATH + "biorxiv_clean.csv")

comm_use_df = pd.read_csv(CLEAN_DATA_PATH + "clean_comm_use.csv")

noncomm_use_df = pd.read_csv(CLEAN_DATA_PATH + "clean_noncomm_use.csv")



# Add all the papers into one large dataframe

papers_df = pd.concat([pmc_df,

                       biorxiv_df,

                       comm_use_df,

                       noncomm_use_df], axis=0).reset_index(drop=True)
# View first few rows of papers dataframe



print (papers_df.shape)

papers_df.head()
# Reading data direct from source



df_cases = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases.csv")

df_cases_country = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")

df_cases_state = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_state.csv")

df_cases_time = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv", parse_dates = ['Last_Update','Report_Date_String'])
# View first few rows of cases data and date of last update



print (df_cases.shape)

print ('Last Update: ' + str(df_cases.Last_Update.max()))

df_cases.head()
# View first few rows of cases by country and date of last update





print (df_cases_country.shape)

print ('Last Update: ' + str(df_cases_country.Last_Update.max()))

df_cases_country.head()
# View first few rows of cases by state and date of last update





print (df_cases_state.shape)

df_cases_state.head()
# View first few rows of cases over time and date of last update



print (df_cases_time.shape)

df_cases_time.head()
# Reading data direct from source



df_confirmed_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df_confirmed_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')

df_deaths_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

df_deaths_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

df_recovered_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
# View first few rows of global confirmed deaths over time and date of last update



print (df_confirmed_global.shape)

df_confirmed_global.head()
# View first few rows of US confirmed deaths over time and date of last update



print (df_confirmed_us.shape)

df_confirmed_us.head()
# View first few rows of global deaths over time and date of last update



print (df_deaths_global.shape)

df_deaths_global.head()
# View first few rows of US deaths over time and date of last update



print (df_deaths_us.shape)

df_deaths_us.head()
# View first few rows of global recoved cases over time and date of last update



print (df_recovered_global.shape)

df_recovered_global.head()