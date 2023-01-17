# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
route = pd.read_csv("../input/coronavirusdataset/route.csv", index_col='id')

time = pd.read_csv("../input/coronavirusdataset/time.csv")

patient = pd.read_csv("../input/coronavirusdataset/patient.csv", index_col='id')
patient.head()
time.head()
route.head()
time['date'] = time['date'].apply(lambda x: '{:%m-%d}'.format(dt.strptime(x, '%Y-%m-%d')))

time.set_index('date', inplace=True)

time.head()
patient.shape
patient.isna().sum()
time.isna().sum()
route.isna().sum()
plt.figure(figsize=(25,6))

sns.set_style("darkgrid")

plt.title("Accumulated Deaths of Coronavirus in South Corea in 2020")

sns.lineplot(data=time['acc_deceased'])
plt.figure(figsize=(25,6))

sns.set_style("darkgrid")



plt.title("Accumulated cases confirmed and deaths of Coronavirus in South Corea in 2020")

sns.lineplot(data=time['acc_confirmed'], label="Cases confirmed")

sns.lineplot(data=time['acc_deceased'], label="Deaths")
patient['age'] = 2020 - patient['birth_year'] 



patient['days_in_hospital'] = pd.to_datetime(patient['released_date'], errors='coerce') - pd.to_datetime(patient['confirmed_date'], errors='coerce') 

patient['days_in_hospital'] = patient['days_in_hospital'].apply(lambda x: x.days)

patient.head()
dead = patient[patient.state == 'deceased']

dead
plt.figure(figsize=(10,6))

sns.set_style("darkgrid")



plt.title("Age distribution of the deceased")

sns.kdeplot(data=dead['age'], shade=True)
male_dead = dead[dead.sex=='male']

female_dead = dead[dead.sex=='female']
plt.figure(figsize=(10,8))

sns.set_style("darkgrid")

sns.distplot(a=male_dead['age'], label="Men", kde=False)

sns.distplot(a=female_dead['age'], label="Women", kde=False)



# Add title

plt.title("Age distribution of the deceased by sex")



# Force legend to appear

plt.legend()



cases_region = pd.DataFrame(patient.groupby(['country','region']).size().reset_index(name='cases'))

cases_region['location'] = cases_region['country'] + ', ' + cases_region['region']

cases_region.drop(columns = ['country', 'region'], inplace=True)

cases_region.sort_values(['cases'], inplace=True, ascending=False)

cases_region.reset_index(drop=True, inplace=True)

cases_region
# Set the width and height of the figure

plt.figure(figsize=(20,6))



# Add title

plt.title("Number of cases by region")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=cases_region.location, y=cases_region.cases)



plt.xticks(rotation='vertical')

# Add label for vertical axis

plt.xlabel("Region")

plt.ylabel("Cases")