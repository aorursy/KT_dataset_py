# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
#warning.filterwarning("ignore")

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
print('Training data shape: ', train.shape)
train.head()

train['winPlacePerc'].plot.hist()
train['walkDistance'].describe()
correlations = train.corr()['winPlacePerc'].sort_values()

print(correlations)
plt.style.use('fivethirtyeight')

plt.hist(train['winPlacePerc'], edgecolor = 'k', bins =25)
plt.title('Walk Distance'); plt.xlabel('Meters'); plt.ylabel('Count');
cor_data = train[['winPlacePerc', 'walkDistance', 'killPlace','boosts','weaponsAcquired']]
cor_data_cors = cor_data.corr()
cor_data_cors
plt.figure(figsize = (8,6))

sns.heatmap(cor_data_cors, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

#polynomial features 

poly_features = train[['walkDistance', 'killPlace','boosts','weaponsAcquired','winPlacePerc']]
poly_features_test = test[['walkDistance', 'killPlace','boosts','weaponsAcquired']]

poly_target = poly_features['winPlacePerc']
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(poly_features)

poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['walkDistance', 'killPlace','boosts','weaponsAcquired'])[:15]
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['walkDistance', 'killPlace','boosts','weaponsAcquired']))

# Add in the target
poly_features['winPlacePerc'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['winPlacePerc'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))