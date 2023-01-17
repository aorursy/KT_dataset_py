# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
rate = pd.read_csv('/kaggle/input/health-insurance-market-plan/Rate_PUF.csv');
rate.head(1)
plan_attributes = pd.read_csv('/kaggle/input/health-insurance-market-plan/Plan_Attributes_PUF.csv');

plan_attributes.head(1)
service_area = pd.read_csv('/kaggle/input/health-insurance-market-plan/Service_Area_PUF.csv');

service_area.head(1)
nan_count_service_area = service_area.isna().sum()

nan_count_plan_attributes = plan_attributes.isna().sum()

nan_count_rate = rate.isna().sum()
print(nan_count_service_area)
temp_service_area = service_area.drop(['PartialCountyJustification','ZipCodes', 'County', 'PartialCounty'], axis=1)

# remove the field that's not so valuable in analysis

temp_service_area = temp_service_area.drop(['BusinessYear', 'ImportDate', 'ServiceAreaId'], axis=1)
service_are = temp_service_area

service_are
attr_to_drop = []

for attr in nan_count_plan_attributes.keys():

    if nan_count_plan_attributes[attr] > 1000:

        attr_to_drop.append(attr)



temp_plan_attributes = plan_attributes.drop(attr_to_drop, axis=1)        

temp_plan_attributes = temp_plan_attributes.drop(['BusinessYear', 'ImportDate', 'TIN', 'ServiceAreaId', 'IsNewPlan'], axis=1)
plan_attributes = temp_plan_attributes

plan_attributes.keys()
attr_to_drop = []

for attr in nan_count_rate.keys():

    if nan_count_rate[attr] > 1000:

        attr_to_drop.append(attr)

temp_rate = rate.drop(attr_to_drop, axis=1)        

temp_rate = temp_rate.drop(['BusinessYear', 'ImportDate', 'RateExpirationDate', 'RateEffectiveDate', 'FederalTIN', 'SourceName'], axis=1)
rate = temp_rate

rate
# Check what are the common fields among these three tables.

set(rate.keys()).intersection(set(plan_attributes.keys()),set(service_area.keys()))
# the kaggle kernel kept restart... I will narrow down the scope.

rate_based_on_provider = rate.drop(['RatingAreaId','PlanId'],axis=1).drop_duplicates()

rate_based_on_provider = rate_based_on_provider[rate_based_on_provider['Age'].str.isnumeric() == True]

rate_based_on_provider
# I want to see how the rate changes based on age

grouped_individual_rate = rate_based_on_provider[['Age', 'IndividualRate','StateCode']].groupby(['StateCode','Age'])

# avg_rate_by_age.plot()

grouped_individual_rate.mean()
grouped_individual_rate.groups.keys()
# Hmm the above result is not what I wanted...

unique_states = rate_based_on_provider.StateCode.unique()

for state in unique_states[:10]:

    avg_age = rate_based_on_provider[rate_based_on_provider['StateCode'] == state][['Age', 'IndividualRate']].groupby(['Age']).mean()

    avg_age.plot(subplots=True, title={state}, grid=True)
sorted_avg_rate_by_state = rate_based_on_provider[['IndividualRate','StateCode']].groupby(['StateCode']).mean().sort_values(by='IndividualRate')
sorted_avg_rate_by_state.plot.bar(figsize=(20, 20))
sorted_avg_rate_by_state.T
# Increasing order of avg insurance rate. WI seems to be the highest

sorted_avg_rate_by_state.T.columns
# Wanna see the trend by age in WI

rate_based_on_provider[rate_based_on_provider['StateCode'] == 'WI'][['Age', 'IndividualRate']].groupby(['Age']).mean().plot(subplots=True, title={state}, grid=True)