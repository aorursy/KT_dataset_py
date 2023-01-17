import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt 

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder

from sklearn import utils 

from sklearn import preprocessing

from sklearn.preprocessing import Imputer

from sklearn.linear_model import LogisticRegression



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train_V2.csv')

test = pd.read_csv('../input/test_V2.csv')
train.info()

test.info()
#converting floats to int as logres can't take in continuous variables 



lab_train = preprocessing.LabelEncoder()

train[['walkDistance']] = lab_train.fit_transform(train[['walkDistance']])

train[['damageDealt']] = lab_train.fit_transform(train[['damageDealt']])

train[['longestKill']] = lab_train.fit_transform(train[['longestKill']])

train[['rideDistance']] = lab_train.fit_transform(train[['rideDistance']])

train[['swimDistance']] = lab_train.fit_transform(train[['swimDistance']])

train[['winPlacePerc']] = lab_train.fit_transform(train[['winPlacePerc']])



test[['walkDistance']] = lab_train.fit_transform(test[['walkDistance']])

test[['damageDealt']] = lab_train.fit_transform(test[['damageDealt']])

test[['longestKill']] = lab_train.fit_transform(test[['longestKill']])

test[['rideDistance']] = lab_train.fit_transform(test[['rideDistance']])

test[['swimDistance']] = lab_train.fit_transform(test[['swimDistance']])



train.info()
#EDA

plt.style.use('fivethirtyeight')

plt.hist(train['walkDistance'], edgecolor = 'k', bins =25)

plt.title('Walk Distance'); plt.xlabel('Meters'); plt.ylabel('Count');

train['walkDistance'].describe()


plt.hist(train['damageDealt'], edgecolor = 'k', bins =25)

plt.title('Damage Dealt'); plt.xlabel('Damage'); plt.ylabel('Count');
train_labels = train['winPlacePerc']



# align the training and testing data, keep only columns present in both dataframes

train, test = train.align(test, join = 'inner', axis = 1)



# add the target back in

train['winPlacePerc'] = train_labels



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)
# correlations 

correlations = train.corr()['winPlacePerc'].sort_values()



print(correlations)
# Checking correlations between each other 

cor_data = train[['winPlacePerc', 'walkDistance', 'killPlace','boosts','weaponsAcquired']]

cor_data_cors = cor_data.corr()

cor_data_cors



plt.figure(figsize = (8,6))

sns.heatmap(cor_data_cors, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)