# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid_path = '../input/bangladesh-covid-19-confirmed-cases-and-death/covid-19_district-wise-quarantine_bangladesh_24.03.2020.xls'
covid_data = pd.read_excel(covid_path,index_col="Division")
covid_data
covid_data.head()
covid_data.tail()
plt.figure(figsize=(20,8))

plt.title("covid19 quarantined cases in various districts of Bangladesh")

sns.barplot(x= covid_data.index, y=covid_data['total_quarantine'])

plt.ylabel("Quarantined cases")
covid1_path = '../input/bangladesh-covid-19-confirmed-cases-and-death/district-wise-confirmed-recovered-cases_06.05.2020.xlsx'
covid1_data = pd.read_excel(covid1_path)
covid1_data
cols_with_missing = [col for col in covid1_data.columns

                     if covid1_data[col].isnull().any()]



# Fill in the lines below: drop columns in training and validation data

reduced_covid1_data =  covid1_data.drop(cols_with_missing, axis=1)

covid1_data
reduced_covid1_data
plt.figure(figsize=(14,7))

plt.title("covid19 Confirmed cases in the month of April and May")

sns.heatmap(data=reduced_covid1_data.tail(), annot=True)

plt.ylabel("Confirmed cases")
cols_with_missing = [col for col in covid_data.columns

                     if covid_data[col].isnull().any()]



# Fill in the lines below: drop columns in training and validation data

reduced_covid_data =  covid_data.drop(cols_with_missing, axis=1)
reduced_covid_data


pd.read_excel(covid_path)
cols_with_missing = [col for col in covid_data.columns

                     if covid_data[col].isnull().any()]



# Fill in the lines below: drop columns in training and validation data

reduced_covid_data =  covid_data.drop(cols_with_missing, axis=1)
sns.scatterplot(x=reduced_covid_data['Shape Area'],y=reduced_covid_data['total_quarantine'])
sns.scatterplot(x=reduced_covid_data['Shape Leng'],y=reduced_covid_data['total_quarantine'])
sns.regplot(x=reduced_covid_data['Shape Area'],y=reduced_covid_data['total_quarantine'])
sns.regplot(x=reduced_covid_data['Shape Leng'],y=reduced_covid_data['total_quarantine'])
reduced_covid1_data.columns
sns.scatterplot(x=reduced_covid1_data['Confirmed_cases Upto 06 May'],y=reduced_covid1_data['Death_cases\nUpto 05 May'])
sns.scatterplot(x=reduced_covid_data['Shape Leng'],y=reduced_covid_data['total_quarantine'],hue=reduced_covid_data['Dist_code'])
sns.scatterplot(x=reduced_covid_data['Shape Area'],y=reduced_covid_data['total_quarantine'],hue=reduced_covid_data['Dist_code'])
sns.swarmplot(x=reduced_covid_data['Shape Area'],

              y=reduced_covid_data['total_quarantine'])
sns.swarmplot(x=reduced_covid_data['Dist_code'],

              y=reduced_covid_data['total_quarantine'])