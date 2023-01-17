# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')

sample = data.head(100)
categorical_sample = sample.iloc[:]['Division Name']+',,'+sample.iloc[:]['Department Name']
def extract_categorical_features(data_features, limit_cat_val=10):

    values_per_feature = {}

    for column in data_features.columns:

        values_per_feature[column] = {}

        for val in data_features.iloc[:][column]:

            if val in values_per_feature[column].keys():

                values_per_feature[column][str(val)]+=1

            else:

                values_per_feature[column][str(val)]=0

    categorical_feature_keys = []

    for key in values_per_feature.keys():

        if len(values_per_feature[key]) < limit_cat_val:

            categorical_feature_keys.append(key)

    categorical_features = []

    for key in categorical_feature_keys:

        categorical_features = np.array(categorical_features) + np.array(data_features.iloc[:][key])

    return categorical_features
cat_features = extract_categorical_features(sample)
values_per_feature = {}

for column in sample.columns:

    values_per_feature[column] = {}

    for val in sample.iloc[:][column]:

        if val in values_per_feature[column].keys():

            values_per_feature[column][str(val)]+=1

        else:

            values_per_feature[column][str(val)]=1
values_per_feature
categorical_features = {}

for feature in values_per_feature.keys():

    print(len(values_per_feature[feature]))

    hasDupli = np.array([val for val in categorical_features[feature].values() if val>2]).any()

    #hasDupli = np.array([val for val in categorical_features[feature].values() if val>2]).any()

    if (len(values_per_feature[feature]) < 20) and hasDupli:

        categorical_features[feature] = values_per_feature[feature]
cat_bool = np.array([val for val in categorical_features['Division Name'].values() if val>60]).any()