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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath="../input/covid19-ourworld/owid-covid-data (6).csv"
covid_data=pd.read_csv(filepath)
pd.set_option('display.max_columns', None)

covid_data=covid_data.drop(columns=['iso_code','total_cases_per_million','new_cases_per_million','total_deaths_per_million','new_deaths_per_million','new_tests_smoothed','new_tests_smoothed_per_thousand','tests_units'])
covid_data['date']=pd.to_datetime(covid_data['date'])
covid_data['Week']=covid_data['date'].dt.weekofyear
covid_data.head()
India=covid_data[covid_data['location']=="India"]


India_weekly1=India.groupby('Week')['new_cases','new_deaths','new_tests',].sum().reset_index()
India_weekly2=India.groupby('Week')['total_cases','total_deaths','total_tests','total_tests_per_thousand'].max().reset_index()
India_weekly3=India.groupby(['Week'])['location','stringency_index','population','population_density','median_age','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','diabetes_prevalence','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_100k'].max().reset_index()

India_weekly=pd.merge(India_weekly1,India_weekly2)
India_weekly=pd.merge(India_weekly,India_weekly3)
India_weekly.head(21)
Country=pd.unique(covid_data['location'])
Country.shape[0]
#print(Country)
np.where(Country=="United States")
my_data1={}
for i in range(Country.shape[0]):
    my_data1[Country[i]]=covid_data[covid_data['location']==Country[i]]
    my_data1[Country[i]]=my_data1[Country[i]].groupby('Week')['new_cases','new_deaths','new_tests',].sum().reset_index()
my_data2={}
for i in range(Country.shape[0]):
    my_data2[Country[i]]=covid_data[covid_data['location']==Country[i]]
    my_data2[Country[i]]=my_data2[Country[i]].groupby('Week')['total_cases','total_deaths','total_tests','total_tests_per_thousand','location','stringency_index','population','population_density','median_age','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','diabetes_prevalence','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_100k'].max().reset_index()
final_data={}
for i in range(Country.shape[0]):
    final_data[Country[i]]=pd.merge(my_data1[Country[i]],my_data2[Country[i]])
final_data[Country[197]]