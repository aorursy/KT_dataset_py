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
state_testing_details = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')

state_testing_details.head()
fig, (axis1) = plt.subplots(1, figsize=(8,8))

sns.barplot(state_testing_details['TotalSamples'],state_testing_details['State'],ax=axis1).set_title('Total Samples of Testing')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(state_testing_details['Negative'],state_testing_details['State'],ax=axis1).set_title('COVID Negative')

sns.barplot(state_testing_details['Positive'],state_testing_details['State'],ax=axis2).set_title('COVID Positive')
fig, (axis1) = plt.subplots(1, figsize=(8,8))

sns.barplot(state_testing_details['Positive']/state_testing_details['TotalSamples']

            ,state_testing_details['State'],ax=axis1).set_title('Fraction of tests giving Positive')
state_testing_details.sort_values(by='Date',ascending=True,inplace=True)

fig, (axis1) = plt.subplots(1, figsize=(8,8))

sns.barplot(state_testing_details['Positive']/state_testing_details['TotalSamples']

            ,state_testing_details['Date'],ax=axis1).set_title('Fraction of tests giving Positive')
population = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')

population.head(2)
population.describe()
population.info()
fig,axes=plt.subplots(1,figsize=(12,6))

sns.barplot(x=population['Urban population']/population['Population'],y=population['State / Union Territory']

            ,ax=axes).set_title('Fraction of urban population')

axes.set_xlabel('Fraction of urban')
plt.figure(figsize=(12,6))

sns.barplot(population['Population'],population['State / Union Territory']).set_title('Population')

plt.xlabel('Population(*10 crore)')
plt.figure(figsize=(12,6))

sns.barplot(population['Rural population'],population['State / Union Territory']).set_title('Rural population')

plt.xlabel('Population(*10 crore)')
plt.figure(figsize=(10,8))

sns.barplot(population['Urban population'],population['State / Union Territory']).set_title('Urban population')

plt.xlabel('Population(*1 crore)')
covid19india=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,8))

sns.barplot(y=covid19india['State/UnionTerritory'],x=covid19india['Deaths'],ax=axis1).set_title('Deaths')

sns.barplot(y=covid19india['State/UnionTerritory'],x=covid19india['Cured'],ax=axis2).set_title('Cured')
covid19india.Date.max()
testing_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

testing_labs.head(2)
labs_states=testing_labs.groupby('state').count().reset_index()[['state','lab']]

labs_states.head()
plt.figure(figsize=(10,8))

sns.barplot(labs_states['lab'],labs_states['state']).set_title('No.of testing centers in States')

plt.xlabel('No. of testing centers')
hospitals = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

hospitals.head(3)
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
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

testing_labs['type'] = labelencoder.fit_transform(testing_labs['type'])

testing_labs.tail()
fig, (axis1) = plt.subplots(1, figsize=(5,8))

sns.barplot(testing_labs['type'],testing_labs['state']).set_title('State VS Testing labs\n (1=GOVT labs & above 1=PVT labs  )')