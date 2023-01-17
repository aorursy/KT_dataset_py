# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.pandas.set_option('display.max_columns', None)
dataset= pd.read_csv('../input/home-data-for-ml-course/train.csv')
print(dataset.shape)
dataset.head()
features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]
print(features_with_na)
for features in features_with_na:
    print(features,np.round(dataset[features].isnull().mean(),4), ' % missing values')
for features in features_with_na:
    data = dataset.copy()
    data[features] = np.where(data[features].isnull(),1,0)
    data.groupby(features)['SalePrice'].median().plot.bar()
    plt.title(features)
    plt.show()
print('id of houses {}'.format(len(dataset.Id)))
numerical_features = [features for features in dataset.columns if dataset[features].dtype != 'O']
print(len(numerical_features))
dataset[numerical_features].head()
year_features = [features for features in numerical_features if 'Yr' in features or 'Year' in features]
year_features
for features in year_features:
    print(features,dataset[features].unique())
for features in year_features:
    if features != 'YrSold':
        data = dataset.copy()
        data[features] = data['YrSold'] - data[features]
        plt.scatter(data[features],data['SalePrice'])
        plt.title(features )
        plt.show()
discrete_features = [features for features in numerical_features if len(dataset[features].unique())<25 and features not in year_features + ['Id'] ]
len(discrete_features)
discrete_features
for features in discrete_features:
    data = dataset.copy()
    data.groupby(features)['SalePrice'].median().plot.bar()
    plt.title(features)
    plt.show()
continous_features = [features for features in numerical_features if features not in discrete_features + year_features + ['Id']]
len(continous_features)
for features in continous_features:
    data = dataset.copy()
    data[features].hist(bins=25)
    plt.title(features)
    plt.show()