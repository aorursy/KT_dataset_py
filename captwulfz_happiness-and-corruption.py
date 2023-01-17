# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math

import matplotlib.pyplot as plt

import numpy as np # linear algebra

from sklearn import datasets, linear_model

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_set16 = pd.read_csv("../input/world-happiness/2016.csv")

data_setCI = pd.read_csv("../input/corruption-index/history.csv")

data_setRR = pd.read_csv("../input/corruption-index/index.csv")
country_setHP = data_set16.loc[data_set16['Country'].isin(data_setCI['Country'])]

country_setCP = data_setCI.loc[data_setCI['Country'].isin(country_setHP['Country'])]

country_setRR = data_setRR.loc[data_setRR['Country'].isin(country_setHP['Country'])]
country_setHP
country_setCP
country_setCP
country_setHP = country_setHP.sort_values('Country')

country_setCP = country_setCP.sort_values('Country')

country_setRR = country_setRR.sort_values('Country')
x = country_setHP['Happiness Score'].values.reshape(146, 1)

y = country_setCP['CPI 2016 Score'].values.reshape(146, 1)



regr = linear_model.LinearRegression()

regr.fit(x, y)



plt.scatter(country_setHP['Happiness Score'], country_setCP['CPI 2016 Score'])

plt.plot(x, regr.predict(x), color = 'red')

plt.xlabel('Happiness Score')

plt.ylabel('Corruption Perception Index')

plt.title('Correlation between Happiness Score and CPI in 2016')



print('Pearson\'s R:')

country_setHP['Happiness Score'].corr(country_setCP['CPI 2016 Score'])
x = country_setHP['Happiness Score'].values.reshape(146, 1)

y = country_setRR['Global Insight Country Risk Ratings'].values.reshape(146, 1)



regr = linear_model.LinearRegression()

regr.fit(x, y)



plt.scatter(country_setHP['Happiness Score'], country_setRR['Global Insight Country Risk Ratings'])

plt.plot(x, regr.predict(x), color = 'red')

plt.xlabel('Happiness Score')

plt.ylabel('Global Risk Ratings')

plt.title('Correlation between Happiness Score and Global Risk Ratings in 2016')



print('Pearson\'s R:')

#print('No Relationship')

country_setHP['Happiness Score'].corr(country_setRR['Global Insight Country Risk Ratings'])