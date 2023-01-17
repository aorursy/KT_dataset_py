# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/waves-measuring-buoys-data-mooloolaba"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
waves = pd.read_csv('../input/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv')
# View the data

waves.head(10)
waves = waves.rename(columns = {'Hs' : 'significant_wave_height' , 'Hmax' : 'maximum_wave_height', 'Tz' : 'zero_wave_period',

                       'Tp' : 'peak_wave_period' , 'SST' : 'sea_surface_temperature' , 'Peak Direction' : 'peak_direction'})
waves.head(10)
# find the variables and Objects on the Data

waves.shape
# Find the type of data in the dataset

waves.info()
waves.isnull().sum()
# Delete First Column

waves1 =waves.drop(columns={'Date/Time'})

waves1.head(10)
waves.describe().transpose()
# Import Libraries

from scipy.stats import skew ,kurtosis
# Find the skewness and Kurtosis on Wave Height Column

print("Skewness of the Waves Height : " ,skew(waves['significant_wave_height']))

print("Kurtosis of the Waves Height : " ,kurtosis(waves['significant_wave_height']))
# Find the skewness and Kurtosis on maximum_wave_height Column

print("Skewness of the maximum_wave_height : " ,skew(waves['maximum_wave_height']))

print("Kurtosis of the maximum_wave_height : " ,kurtosis(waves['maximum_wave_height']))
# Find the skewness and Kurtosis on zero_upcrossing_wave_period Column

print("Skewness of the zero_upcrossing_wave_period : " ,skew(waves['zero_wave_period']))

print("Kurtosis of the zero_upcrossing_wave_period : " ,kurtosis(waves['zero_wave_period']))
# Find the skewness and Kurtosis on peak_energy_wave_period Column

print("Skewness of the peak_energy_wave_period : " ,skew(waves['peak_wave_period']))

print("Kurtosis of the peak_energy_wave_period : " ,kurtosis(waves['peak_wave_period']))
# Find the skewness and Kurtosis on Peak Direction Column

print("Skewness of the Peak Direction : " ,skew(waves['peak_direction']))

print("Kurtosis of the Peak Direction : " ,kurtosis(waves['peak_direction']))
# Find the skewness and Kurtosis on sea_surface_temperature Column

print("Skewness of the sea_surface_temperature : " ,skew(waves['sea_surface_temperature']))

print("Kurtosis of the sea_surface_temperature : " ,kurtosis(waves['sea_surface_temperature']))
# Import Libraries

import matplotlib.pyplot as plt

import seaborn as sns
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(waves['significant_wave_height'])

plt.show()
waves['significant_wave_height'].hist()
sns.boxplot(waves['significant_wave_height'])
plt.figure(figsize=(10,6)) 

plt.title("significant_wave_height") 

sns.kdeplot(data=waves['significant_wave_height'], label="significant_wave_height", shade=True)
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(waves['maximum_wave_height'])

plt.show()
waves['maximum_wave_height'].hist()
sns.boxplot(waves['maximum_wave_height'])
plt.figure(figsize=(10,6)) 

plt.title("maximum_wave_height") 

sns.kdeplot(data=waves['maximum_wave_height'], label="maximum_wave_height", shade=True)
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(waves['zero_wave_period'])

plt.show()
waves['zero_wave_period'].hist()
sns.boxplot(waves['zero_wave_period'])
plt.figure(figsize=(10,6)) 

plt.title("zero_wave_period") 

sns.kdeplot(data=waves['zero_wave_period'], label="zero_wave_period", shade=True)
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(waves['peak_wave_period'])

plt.show()
waves['peak_wave_period'].hist()
sns.boxplot(waves['peak_wave_period'])
plt.figure(figsize=(10,6)) 

plt.title("peak_wave_period") 

sns.kdeplot(data=waves['peak_wave_period'], label="peak_wave_period", shade=True)
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(waves['peak_direction'])

plt.show()
waves['peak_direction'].hist()
sns.boxplot(waves['peak_direction'])
plt.figure(figsize=(10,6)) 

plt.title("Peak Direction") 

sns.kdeplot(data=waves['peak_direction'], label="Peak Direction", shade=True)
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(waves['sea_surface_temperature'])

plt.show()
waves['sea_surface_temperature'].hist()
sns.boxplot(waves['sea_surface_temperature'])
plt.figure(figsize=(10,6)) 

plt.title("sea_surface_temperature") 

sns.kdeplot(data=waves['sea_surface_temperature'], label="sea_surface_temperature", shade=True)
# import libraries

from sklearn.preprocessing import scale

scale(waves1)
np.exp(scale(waves1))
sns.pairplot(waves)
plt.figure(figsize=(14,10))

sns.heatmap(waves.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)

plt.show()
waves.corr()
# Prepare Regression Model using all Objects

# import libraries 

import statsmodels.formula.api as smf
# Preparing model                  

Regression = smf.ols('significant_wave_height~maximum_wave_height+zero_wave_period+peak_wave_period+sea_surface_temperature+peak_direction',data=waves).fit() # regression model
# Getting coefficients of variables               

Regression.params
# Summary

Regression.summary()
# Import Libraries

from pandas import DataFrame

from sklearn import linear_model

import statsmodels.api as sm
x = waves1.drop(['significant_wave_height'], axis = 1)

y = waves1.significant_wave_height.values
# Fit the Model

regr = linear_model.LinearRegression()

regr.fit(x, y)
print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)
# Build Model

model = sm.OLS(y, x).fit()

predictions = model.predict(x) 
print_model = model.summary()

print(print_model)