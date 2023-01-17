# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from glob import glob

import matplotlib.pyplot as plt



filenames = glob('/kaggle/input/novel-corona-virus-2019-dataset/*')

filenames.sort()

filenames

# Any results you write to the current directory are saved as output.
#read all the files

line_list_data = pd.read_csv(filenames[0])

open_line_list = pd.read_csv(filenames[1])

data = pd.read_csv(filenames[2])

confirmed = pd.read_csv(filenames[3])

deaths = pd.read_csv(filenames[4])

recovered = pd.read_csv(filenames[5])
confirmed_count = sum(confirmed.iloc[:,-1]) #confirmed cases

death_count = sum(deaths.iloc[:,-1]) #death cases

recovered_count = sum(recovered.iloc[:,-1]) #recovered cases



print('Number of confirmed cases: ', confirmed_count)

print('Number of recovered cases: ', recovered_count)

print('Number of death cases: ', death_count)

print('Number of active cases: ', confirmed_count - recovered_count - death_count)
#extract the required columns and calculate the total number of cases per day

confirmed_cases_datewise = confirmed.iloc[:,4:]

confirmed_cases = [sum(confirmed_cases_datewise[column]) for column in confirmed_cases_datewise.columns]

confirmed_cases.insert(0, 0)

cases_per_day = [(col, confirmed_cases[ind+1] - confirmed_cases[ind]) for ind, col in enumerate(confirmed_cases_datewise.columns)]



#calculate number of cases per week from cases per day data

cases_per_week = {}

for ind in range(0, len(cases_per_day), 7):

    key = cases_per_day[ind][0]

    try:

        value = sum([cases_per_day[x][1] for x in range(ind, ind+7, 1)])

    except:

        value = sum([cases_per_day[x][1] for x in range(ind, len(cases_per_day), 1)])

    cases_per_week[key] = value



#bar graph

plt.bar(cases_per_week.keys(), cases_per_week.values(), width=0.8)

plt.xticks(rotation=45)

plt.show()