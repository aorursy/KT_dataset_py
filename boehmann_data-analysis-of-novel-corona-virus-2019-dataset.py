# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Add Cororna Datasets 

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
# first overview

COVID19_line_list_data.head(2)
#number of entris by column

COVID19_line_list_data.count()
# which countries are in the set?

pd.unique(COVID19_line_list_data.country)
# List all Columns from dataset

COVID19_line_list_data.columns
# what is in the unnamed columns?

COVID19_line_list_data.dtypes
# clear the symptoms

covid19_symptoms = COVID19_line_list_data.symptom

#covid19_symptoms.dropna()

print('number of unique symptoms: \n',covid19_symptoms.value_counts())

# config of PLots

import matplotlib.pyplot as plt

print(plt.rcParams.get('figure.figsize'))

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 15

fig_size[1] = 6

plt.rcParams["figure.figsize"] = fig_size
#visualize symptoms in bar plot



covid19_symptoms_numbers = COVID19_line_list_data.symptom.value_counts().tolist()

covid19_symptoms_names =  COVID19_line_list_data.symptom.value_counts().index.tolist()



# distindt values smaller 2 in list for more relevant information in plot

covid19_symptoms_numbers = [item for item in covid19_symptoms_numbers if item > 2 ]

# show symptoms over the count of 2

print(covid19_symptoms_numbers)

y_pos = np.arange(len(covid19_symptoms_numbers))



# Barplot of the symptoms

plt.bar(y_pos, covid19_symptoms_numbers, color=(0.2, 0.4, 0.6, 0.6))

plt.xticks(y_pos, covid19_symptoms_names)

degrees = 70

plt.xticks(rotation=degrees)

plt.show()
# config of PLots

print(plt.rcParams.get('figure.figsize'))

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 8

fig_size[1] = 6

plt.rcParams["figure.figsize"] = fig_size





COVID19_line_list_data.groupby('gender').size().plot(kind='bar')

print(COVID19_line_list_data.gender.value_counts())



# percentile 

print(COVID19_line_list_data.gender.value_counts().sum())

COVID19_sum = COVID19_line_list_data.gender.value_counts().sum() / 100

COVID19_male = sum(COVID19_line_list_data.gender == 'male')

COVID19_female = sum(COVID19_line_list_data.gender == 'female')



COVID19_male_percent = COVID19_male / COVID19_sum 

COVID19_female_percent = COVID19_female / COVID19_sum 



print('percent distribution of the gender male: ', COVID19_male_percent, '%')

print('percent distribution of the gender female: ', COVID19_female_percent, '%')