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
raw=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
raw.describe()
import seaborn as sns

sns.distplot(raw.price)
sns.scatterplot(raw.id,raw.price)
sns.scatterplot(raw.longitude,raw.latitude,hue=raw.neighbourhood_group)
np.sum(raw.isna())
sns.heatmap(raw.corr())
data=raw
data.neighbourhood_group.value_counts()
data.room_type.value_counts()
data.reviews_per_month.fillna(0,inplace=True)
data['dt']=pd.to_datetime(data.last_review)

today = pd.to_datetime('2019-8-28')

data['gap']=today - data.dt

data['gap']= data['gap'] / np.timedelta64(1, 'D')

data.gap.fillna(1000,inplace=True)
feature=['host_id','neighbourhood_group',

       'neighbourhood', 'latitude', 'longitude', 'room_type',

       'minimum_nights', 'number_of_reviews', 'gap',

       'reviews_per_month', 'calculated_host_listings_count',

       'availability_365','price']

X=data[feature]

y=data['number_of_reviews']
from sklearn.preprocessing import LabelEncoder

X[['neighbourhood_group', 'neighbourhood', 'room_type']]=X[['neighbourhood_group', 'neighbourhood', 'room_type']].apply(LabelEncoder().fit_transform)
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
import lightgbm as lgb

from sklearn import metrics



categorical_feature = ['host_id', 'neighbourhood_group', 'neighbourhood', 'room_type']

train = lgb.Dataset(data=X_train,label=y_train,categorical_feature=categorical_feature)

test = lgb.Dataset(data=X_test,label=y_test,categorical_feature=categorical_feature)



param = {'num_leaves':31, 'num_trees':1000, 'objective':'regression'}

param['metric'] = 'rmse'

param['num_threads'] = 2

param['early_stopping_round'] = 50

param['ignore_column'] =  3,4



bst = lgb.train(param, train, valid_sets=[train,test])