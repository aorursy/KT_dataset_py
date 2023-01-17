# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#load dataset

x = pd.read_csv('../input/survey_results_public.csv')
#cleaning data

cleanup_js = {"JobSatisfaction": {"Extremely satisfied": 3, "Moderately satisfied": 2, "Extremely dissatisfied": -3, "Moderately dissatisfied": -2, "Slightly satisfied": 1, "Slightly dissatisfied": -1, "Neither satisfied nor dissatisfied": 0},
             "CareerSatisfaction": {"Extremely satisfied": 3, "Moderately satisfied": 2, "Extremely dissatisfied": -3, "Moderately dissatisfied": -2, "Slightly satisfied": 1, "Slightly dissatisfied": -1, "Neither satisfied nor dissatisfied": 0},
             "Exercise": {"I don't typically exercise": 0, "1 - 2 times per week": 1, "3 - 4 times per week": 2, "Daily or almost every day": 3},
             "HoursOutside": {"Less than 30 minutes": 0, "30 - 59 minutes": 1, "1 - 2 hours": 2, "3 - 4 hours": 3, "Over 4 hours": 4},
             "SkipMeals": {"Never": 1, "1 - 2 times per week": -1, "3 - 4 times per week": -2, "Daily or almost every day": -3}}

x.replace(cleanup_js, inplace=True)
#drop row with NaN

x.dropna(subset=['JobSatisfaction', 'Exercise', 'SkipMeals', 'HoursOutside', 'CareerSatisfaction'], how='any', inplace=True)
#fillna in ErgonomicDevices

x.ErgonomicDevices = x.ErgonomicDevices.fillna("no")
#calculates ErgonomicDevices points

x['ErgPoints'] = np.where(x['ErgonomicDevices'] == 'no', 0, x['ErgonomicDevices'].str.count(';') + 1)
#data aggregation

x['TotSatisfaction'] = x['JobSatisfaction'] + x['CareerSatisfaction']

x['TotWellbeing'] = x['Exercise'] + x['HoursOutside'] + x['SkipMeals'] +x['ErgPoints']
#show the table

x.loc[1:100, x.columns.isin(list(['ErgPoints', 'TotWellbeing', 'TotSatisfaction', 'SkipMeals', 'ErgonomicDevices', 'Exercise', 'HoursOutside', 'JobSatisfaction', 'CareerSatisfaction', 'Gender']))]
#normalize TotSatisfaction points

x['TotSatisfactionNorm'] = [float(i)/max(x['TotSatisfaction']) for i in x['TotSatisfaction']]
print(x['TotSatisfactionNorm'])
#normalize TotWellbeing points

x['TotWellbeingNorm'] = [float(i)/max(x['TotWellbeing']) for i in x['TotWellbeing']]
print(x['TotWellbeingNorm'])
#show the table

x.loc[1:100, x.columns.isin(list(['TotWellbeingNorm', 'TotSatisfactionNorm', 'Gender']))].sort_values(by='TotWellbeingNorm', ascending=False)
#show plot

import seaborn as sns

sns.kdeplot(x['TotWellbeingNorm'])
sns.kdeplot(x['TotSatisfactionNorm'])
#show graph_hist

plt.hist(x['TotWellbeingNorm'], bins=50, histtype='stepfilled', normed=True, color='b', label='Wellbeing')
plt.hist(x['TotSatisfactionNorm'], bins=50, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Satisfaction')
plt.title("Gaussian/Uniform Histogram")
plt.xlabel("Points")
plt.ylabel("Frequencies")
plt.legend()
plt.show()
#test spearman correlation

data = x[['TotSatisfactionNorm','TotWellbeingNorm', 'JobSatisfaction', 'CareerSatisfaction', 'HoursOutside', 'SkipMeals', 'Exercise', 'TotWellbeing', 'TotSatisfaction']]
correlation = data.corr(method='spearman')
print(correlation)
#test pearson correlation

data = x[['TotSatisfactionNorm','TotWellbeingNorm', 'JobSatisfaction', 'CareerSatisfaction', 'HoursOutside', 'SkipMeals', 'Exercise', 'TotWellbeing', 'TotSatisfaction']]
correlation = data.corr(method='pearson')
print(correlation)