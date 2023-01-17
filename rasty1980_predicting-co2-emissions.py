import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Machine Learning

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import xgboost as xgb

import lightgbm as lgb

#from sklearn import cross_validation

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, classification_report

from scipy.stats import norm, skew

from statistics import mode



import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

    

import os

print(os.listdir("../input"))

data = pd.read_csv('../input/Car_Fuel_Consumption.csv')
data.head()
data['ID'] = data.index + 1

cols = data.columns.tolist()

cols = cols[-1:] + cols[:-1]

data = data[cols]
#Getting the missing % count

missingData = (data.isnull().sum() / len(data)) * 100

missingData = pd.DataFrame({'Missing Percentage' :missingData})

missingData =  missingData.sort_values(by='Missing Percentage', ascending=False)

missingData['Missing Percentage'] = missingData['Missing Percentage'].round(4)

missingData.head(25)
data.columns
colsToRemove = ['model','description','transmission','fuel_cost_12000_miles','fuel_cost_6000_miles', 'standard_12_months', 

                'standard_6_months','first_year_12_months', 'first_year_6_months', 'date_of_change']

data = data.drop(colsToRemove, axis=1)

#list(data)
#Checking the Categorical Features

data.dtypes.sample(16)
# Checking for skewness

numeric_feats = data.dtypes[data.dtypes != "object"].index



# Compute skewness

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(13)
#Graphing boxplots to understand the skewness

features =['extra_urban_metric','engine_capacity' ]

#fig, ax =plt.subplots(1,2)

plt.figure(figsize=(4,4))

sns.boxplot(data['extra_urban_metric'])

plt.show()

plt.figure(figsize=(4,4))

sns.boxplot(data['noise_level'])

plt.show()

plt.figure(figsize=(4,4))

sns.boxplot(data['engine_capacity'])

plt.show()

#fig.show()
#Dropping Rows with outliers

data = data.drop(data[data.extra_urban_metric > 80].index)

data = data.drop(data[data.noise_level < 40].index)
# Checking for skewness AGAIN

numeric_feats = data.dtypes[data.dtypes != "object"].index



# Compute skewness

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(13)
#Log transformig the skewed variables ready for Modeling

skewed_features = ['engine_capacity', 'co_emissions','urban_metric','combined_metric','co2','extra_urban_metric' ]

#skewed_features = skewness.index

for feature in skewed_features:

    data[feature] = np.log1p(data[feature])
# Checking for skewness AGAIN

numeric_feats = data.dtypes[data.dtypes != "object"].index



# Compute skewness

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(13)
#Filtering Fuel Type in order to graph only significant types

fuelType = ['Petrol', 'Diesel', 'LPG', 'Petrol Hybrid']

#topFourFuel = data.fuel_type.isin(fuelType)

topFourFuel= data[data.fuel_type.isin(fuelType)]
#plt.figure(figsize=(15,10))

custom_palette = sns.color_palette("Set1", 2)

#sns.palplot(custom_palette)

g = sns.FacetGrid(data=topFourFuel, col='fuel_type',hue='transmission_type', col_wrap=4, height=5, palette = custom_palette)

g = (g.map(sns.scatterplot, 'engine_capacity','co2' ).add_legend())
#plottind the co2 distributions ba

g = sns.FacetGrid(topFourFuel, col="fuel_type", col_wrap=4,height=4, palette=custom_palette)

g = (g.map(sns.distplot, "co2").add_legend())
#topFourFuel = topFourFuel.drop('ID', axis=1)

meanTopFourFuel = topFourFuel.groupby('fuel_type').mean()

meanTopFourFuel = meanTopFourFuel.round(2)

meanTopFourFuel
ohe = pd.get_dummies(data)

#ohe.head().T
y = ohe.co2 # this is the variable that i will be attempting to predict
# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(ohe, y, test_size=0.2, random_state = 8)



#Saving the ID feauture

train_ID = X_train['ID']

test_ID = X_test['ID']



#dropping ID from data sets as they are unnecesary

X_train = X_train.drop('ID', axis = 1)

X_test = X_test.drop('ID', axis=1)



#saving the train and test row shapes

ntrain = X_train.shape[0]

ntest = X_test.shape[0]



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
mod1 = xgb.XGBRegressor(gamma=0, objective='reg:linear', nthread=-1)

mod1.fit(X_train, y_train)

y_pred = mod1.predict(X_test)

print('score for gridsearch:' , mean_squared_error(y_test, y_pred))

#data.co2[26337]

#y_train

#train_ID

#X_train

#X_test

#ntrain

#ntest

data[data['extra_urban_metric']>5]
ohe = pd.get_dummies(data)
ohe.head().T