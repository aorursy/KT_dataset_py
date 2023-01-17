# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
country_data=data[['Country','Confirmed','Deaths','Recovered']].groupby('Country').sum().reset_index()

country_not_china=country_data[(country_data['Country']!='Mainland China') & (country_data['Country']!='Others') & (country_data['Country']!='China') & (country_data['Country']!='Vietnam')]
plt.figure(figsize=(12,8))

chart=sns.barplot(country_not_china['Country'],country_not_china['Confirmed'])

plt.xticks(rotation=60)

plt.title("No. of Confirmed cases of Corona virus")
plt.figure(figsize=(12,8))

chart=sns.barplot(country_not_china['Country'],country_not_china['Deaths'])

plt.xticks(rotation=60)

plt.title("No. of Confirmed deaths due to Corona virus")
plt.figure(figsize=(12,8))

chart=sns.barplot(country_not_china['Country'],country_not_china['Recovered'])

plt.xticks(rotation=60)

plt.title("No. of patients recovered/cured from Corona virus")
print("Please enter Mainland China for China")

data_confirmed=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

country_confirmed=data_confirmed.groupby('Country/Region').sum().reset_index().drop(['Lat', 'Long'], axis=1)



country=input('Enter Country name to be view day to day details: ')

country_list=list(country_confirmed['Country/Region'])

if(country not in country_list):

    print("Country not found.")

else:

    plt.figure(figsize=(15,5))

    country_specific=country_confirmed[country_confirmed['Country/Region']==country].drop(['Country/Region'],axis=1).T

    chart=sns.lineplot(data=country_specific,markers=["D"],label="Confirmed Cases",legend=False)

    

    data_deaths=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

    country_deaths=data_deaths.groupby('Country/Region').sum().reset_index().drop(['Lat', 'Long'], axis=1)

    country_specific=country_deaths[country_deaths['Country/Region']==country].drop(['Country/Region'],axis=1).T

    chart=sns.lineplot(data=country_specific,markers=["X"],label="Deaths",legend=False, palette=['red'],linewidth=2)



    data_recovered=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

    country_recovered=data_recovered.groupby('Country/Region').sum().reset_index().drop(['Lat', 'Long'], axis=1)

    country_specific=country_recovered[country_recovered['Country/Region']==country].drop(['Country/Region'],axis=1).T

    chart=sns.lineplot(data=country_specific,markers=True,label="Cured Cases",legend=False,palette=['green'])

    plt.xticks(rotation=30)

    plt.legend()