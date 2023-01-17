# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw_cc_data = pd.read_csv('../input/creditcard.csv').dropna()
raw_cc_data.info()
raw_cc_data.head()
# get how many (non) frauds there are
print ('There are {} non fraudulent charges and {} fraudulent charges.'.format(len(raw_cc_data[raw_cc_data['Class'] == 0]), len(raw_cc_data[raw_cc_data['Class'] == 1])))
# data is highly skewed, keep data of fradulent charges, but use subsample of non fraudulent charges to create an oversampling of fradulent charges 
fraud_data = raw_cc_data[raw_cc_data['Class']==1]
nonfraud_data = raw_cc_data[raw_cc_data['Class']==0].sample(n=250)

sample_data = pd.concat([fraud_data, nonfraud_data], ignore_index=True)
# standardize the data
x = sample_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scale_data = pd.DataFrame(x_scaled, columns=sample_data.columns)
g = plt.figure(dpi=100, figsize=(9,9))
g = sns.heatmap(scale_data.corr(), xticklabels=True, yticklabels=True, square=True, vmax=.9, center=0.5)
# v1, v3, v5, v7 are similar
# v12, v14, v16, v17, v18 are similar
# v9, v10 are similar
print(scale_data[['V1', 'V3', 'V5', 'V7']].corr())
print(scale_data[['V12', 'V14', 'V16', 'V17', 'V18']].corr())
print(scale_data[['V9', 'V10']].corr())
# combine V1, V3, V5, V7
# combine V9, V10
# combine V12, V14
# combine V16, V17, V18
features = pd.DataFrame(scale_data)
features['V1-3-5-7'] = features[['V1', 'V3', 'V5', 'V7']].sum(1)/4
features['V9-10'] = features[['V9', 'V10']].sum(1)/2
features['V12-14'] = features[['V12', 'V14']].sum(1)/2
features['V16-17-18'] = features[['V16', 'V17', 'V18']].sum(1)/3
features.drop(['V1', 'V3', 'V5', 'V7', 'V9', 'V10', 'V12', 'V14', 'V16', 'V17', 'V18'], 1, inplace=True)
g = plt.figure(dpi=100, figsize=(9,9))
g = sns.heatmap(features.corr(), xticklabels=True, yticklabels=True, square=True, vmax=.9, center=0.5)
features
# check accuracy rate for fradulent charges
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# fit model to data; exclude no correlations
y = features.Class
x = features.loc[:, ~features.columns.isin(['Class', 'Time', 'V8', 'V13', 'V15', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])]

# Create new training Set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

bnb.fit(x_train, y_train)

#classify, store results in variable
y_pred = bnb.predict(x_test)

#display
print("number of mislabeled points out of total {} points: {}; {:.2f}% accurate".format(
    x_test.shape[0], (y_test != y_pred).sum(), ((1-y_test != y_pred).sum()/x_test.shape[0] * 100)
))
y_test = y_test.values.reshape(-1, 1)
count, total = [0, 0]

for i in range(len(y_test)):
    if (y_test[i]==1):
        if (y_pred[i] == 1):
            count = count + 1
        total = total + 1
print(count, 'fraudulent charges correctly identified out of a total of ', total, 'fraudulent charges')