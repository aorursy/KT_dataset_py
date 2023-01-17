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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
population = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')

population.tail()
fig, (axis1) = plt.subplots(1,figsize=(10,8))

sns.barplot(population['Population'],population['State / Union Territory'],ax=axis1).set_title('Population')
fig, (axis1) = plt.subplots(1,figsize=(10,8))

sns.barplot(population['Rural population'],population['State / Union Territory'],ax=axis1).set_title('Rural population')
fig, (axis1) = plt.subplots(1,figsize=(10,8))

sns.barplot(population['Urban population'],population['State / Union Territory']).set_title('Urban population')
covid19india = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

covid19india['Date'] = pd.to_datetime(covid19india.Date)

covid19india.head()
covid19india['Day'] = covid19india.Date.dt.day

covid19india['Month'] = covid19india.Date.dt.month

covid19india.head()
hospitals = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

hospitals.tail()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(hospitals['NumCommunityHealthCenters_HMIS'],hospitals['State/UT'],ax=axis1).set_title('Num of Community HealthCenters')

sns.barplot(hospitals['NumSubDistrictHospitals_HMIS'],hospitals['State/UT'],ax=axis2).set_title('Num of SubDistrict Hospitals')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(hospitals['TotalPublicHealthFacilities_HMIS'],hospitals['State/UT'],ax=axis1).set_title('Total Public Health Facilities')

sns.barplot(hospitals['NumPublicBeds_HMIS'],hospitals['State/UT'],ax=axis2).set_title('Num of Public Beds')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(hospitals['NumRuralHospitals_NHP18'],hospitals['State/UT'],ax=axis1).set_title('Num of Rural Hospitals')

sns.barplot(hospitals['NumRuralBeds_NHP18'],hospitals['State/UT'],ax=axis2).set_title('Num of Rural Beds')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(hospitals['NumUrbanHospitals_NHP18'],hospitals['State/UT'],ax=axis1).set_title('Num of Urban Hospitals')

sns.barplot(hospitals['NumUrbanBeds_NHP18'],hospitals['State/UT'],ax=axis2).set_title('Num of Urban Beds')
testing_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

testing_labs.head()
state_testing_details = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')

state_testing_details.head()
fig, (axis1) = plt.subplots(1, figsize=(8,8))

sns.barplot(state_testing_details['TotalSamples'],state_testing_details['State'],ax=axis1).set_title('Total Samples of Testing')
fig, (axis1) = plt.subplots(1, figsize=(5,8))

sns.barplot(y=covid19india['State/UnionTerritory'],x=covid19india['Cured']).set_title('Cured')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(state_testing_details['Negative'],state_testing_details['State'],ax=axis1).set_title('COVID Negative')

sns.barplot(state_testing_details['Positive'],state_testing_details['State'],ax=axis2).set_title('COVID Positive')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(y=covid19india['State/UnionTerritory'],x=covid19india['Deaths'],ax=axis1).set_title('Deaths')

sns.barplot(y=covid19india['State/UnionTerritory'],x=covid19india['Cured'],ax=axis2).set_title('Cured')
testing_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

testing_labs.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

testing_labs['type'] = labelencoder.fit_transform(testing_labs['type'])

testing_labs.tail()
fig, (axis1) = plt.subplots(1, figsize=(5,8))

sns.barplot(testing_labs['type'],testing_labs['state']).set_title('State VS Testing labs\n (1=GOVT labs & above 1=PVT labs  )')
individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

individual_details.head()
individual_details['diagnosed_date'] = pd.to_datetime(individual_details.diagnosed_date)

individual_details['diagnosed_month'] = individual_details.diagnosed_date.dt.month

individual_details['diagnosed_day'] = individual_details.diagnosed_date.dt.day

individual_details.head(2)
individual_details['status_change_date'] = pd.to_datetime(individual_details.status_change_date)

individual_details['status_change_month'] = individual_details.status_change_date.dt.month

individual_details['status_change_day'] = individual_details.status_change_date.dt.day

individual_details.head()
fig, (axis1) = plt.subplots(1, figsize=(8,6))

sns.barplot(individual_details['status_change_month'],individual_details['diagnosed_month'],hue=individual_details['current_status']).set_title('Diagnosed month VS Status change month VS Current status\n(january=1 & December=12 )')