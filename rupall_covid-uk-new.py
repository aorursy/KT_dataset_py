# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/covid-cases-uk'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import os
for dirname, _, filenames in os.walk('../input/covid-deaths-uk'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import os
for dirname, _, filenames in os.walk('../input/covid-hospital-uk'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

import os
for dirname, _, filenames in os.walk('../input/covid-testing-uk'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing cases csv file.
covid_cases_df = pd.read_csv('../input/covid-cases-uk/UK_data_2020-Oct-11.csv', 
                             usecols=['date','newCasesBySpecimenDate' ])
covid_cases_df.columns = ['date', 'cases']
covid_cases_df.head()
# to view the entire dataframe
pd.set_option('display.max_rows', covid_cases_df.shape[0]+1)
print(covid_cases_df)
covid_cases_df.shape
covid_cases_df.nunique()
covid_cases_df.info()
covid_cases_df.loc[971:975]
# sorting the df to check the starting date reported
covid_cases_df.sort_values('date',ascending=True).head()
# sorting the df to check the end date reported
covid_cases_df.sort_values('date',ascending=True).tail()
covid_cases_df.info()
# Info shows that the data type of date is objcet.
# convert date data type
covid_cases_df['date']= pd.to_datetime(covid_cases_df.date)
covid_cases_df.info()
covid_cases_df.date.dt.day
covid_cases_df.date.duplicated()
covid_cases_df.loc[970:974,:]
# Searching multiple date entries in covid dataframe.
ts =pd.to_datetime('2020-10-10')
covid_cases_df.loc[covid_cases_df.date == ts,:]
# Searching multiple date entries in covid dataframe.
ts =pd.to_datetime('2020-02-28')
covid_cases_df.loc[covid_cases_df.date == ts,:]
groupby_object = covid_cases_df.groupby('date').cases.sum()
groupby_object
groupby_object.describe()
groupby_object.shape
groupby_object.plot()

groupby_object.to_frame().reset_index()

# importing death_cases csv file from data.gov.uk website.

covid_deaths_df = pd.read_csv('../input/covid-deaths-uk/uk_death_data_2020-Oct-11 (1).csv',usecols=['date','newDeaths28DaysByDeathDate' ])
covid_deaths_df.columns = ['date', 'deaths']
covid_deaths_df
covid_deaths_df.shape
covid_deaths_df.date.nunique()
#info shows the date data type is object
covid_deaths_df.info()
# converting date data type 
covid_deaths_df['date']= pd.to_datetime(covid_deaths_df.date)
covid_deaths_df.info()
covid_deaths_df.date.max()- covid_deaths_df.date.min()
death_object = covid_deaths_df.groupby('date').deaths.sum()
death_object
death_object.describe()
death_object.plot()
merged_cd = pd.merge(groupby_object, death_object, on='date', how='left')
merged_cd.tail(20)
merged_cd.shape
merged_cd.describe()
merged_cd.plot()
covid_deaths_df['Month']= covid_deaths_df.date.dt.month
covid_deaths_df.head()
covid_deaths_df.Month.value_counts().sort_index().plot()