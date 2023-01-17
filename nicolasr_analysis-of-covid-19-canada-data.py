# ANALYSIS OF THE 'covid_19_canada_open_data_working_group'data.
# SCROLL TO END FOR CONCLUSION
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
dataframe = pd.read_csv('/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/public-covid-19-cases-canada.csv')
dataframe.head()
import matplotlib.pyplot as plt 
%matplotlib inline
# What's the age of the patient?
dataframe['age'].value_counts()
# Are there more cases among people who have travelled recently?
dataframe['has_travel_history'].value_counts()
# What province has the most cases currently?
dataframe['province'].value_counts()
# What health region has the most cases currently?
healthregion = dataframe.groupby(['health_region']).size().sort_values(ascending=False)
pd.set_option('display.max_rows', healthregion.shape[0]+1)
print(healthregion.head(30))
# What is the sex of the patient?
dataframe['sex'].value_counts()
# Now we analyze the patients with reported age in order to learn
# more about the age and sex of those patients.
# Also: What province has the biggest group of patients?

# For this we remove all entries without reported age
#Select the entries "Not Reported" of the 'age'-column
indexNames = dataframe[ dataframe['age'] == 'Not Reported' ].index

#Delete these entries from the dataframe
dataframe.drop(indexNames , inplace=True)
# Most affected patients by age (only entries with reported age)
dataframe['age'].value_counts()
# Are patients with travel history affected more
# than patients without travel history? (based on patients with recorded age)
dataframe['has_travel_history'].value_counts()
# Total male patients with reported age
totalMales = dataframe[dataframe['sex'].str.contains('Male')]
totalMales['sex'].value_counts()
# Total female patients with reported age
totalMales = dataframe[dataframe['sex'].str.contains('Female')]
totalMales['sex'].value_counts()
# Most affected patients with reported age sorted by
# 1) sex
# 2) has travel history
travelhistory = dataframe.groupby(['sex','has_travel_history']).size().sort_values(ascending=False)
pd.set_option('display.max_rows', travelhistory.shape[0]+1)
print(travelhistory.head(30))
# Are there more male or female patients in one specific age group?

age = dataframe.groupby(['age', 'sex']).size().sort_values(ascending=False)
pd.set_option('display.max_rows', age.shape[0]+1)
print(age.head(30))