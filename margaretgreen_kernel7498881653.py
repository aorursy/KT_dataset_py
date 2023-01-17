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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/berlin-airbnb-prices/berlin_airbnb_train.csv')

kek = pd.read_csv('/kaggle/input/berlin-airbnb-prices/berlin_airbnb_test.csv')

sam = pd.read_csv('/kaggle/input/berlin-airbnb-prices/submit.csv')
train.info()
np.unique(train.room_type)
room_dict = {'Entire home/apt': 3, 'Private room': 2, 'Shared room': 1}

train.room_type = train.room_type.replace(to_replace=room_dict)

kek.room_type = kek.room_type.replace(to_replace=room_dict)
np.unique(train.bed_type)
bed_dict = {'Real Bed': 5, 'Pull-out Sofa': 4, 'Couch': 3, 'Airbed': 2, 'Futon': 1}

train.bed_type = train.bed_type.replace(to_replace=bed_dict)

kek.bed_type = kek.bed_type.replace(to_replace=bed_dict)
np.unique(train.cancellation_policy)
can_dict = {'flexible': 5, 'moderate': 4, 'strict_14_with_grace_period': 3, 'super_strict_30': 2, 'super_strict_60': 1}

train.cancellation_policy = train.cancellation_policy.replace(to_replace=can_dict)

kek.cancellation_policy = kek.cancellation_policy.replace(to_replace=can_dict)
tar = 'price'
feature_columns = ['accommodates', 'bathrooms', 'bedrooms', 'price', 'cleaning_fee', 'security_deposit', 'extra_people', 'guests_included', 'distance', 'size', 'room_type', 'bed_type', 'minimum_nights', 'cancellation_policy', 'Laptop_friendly_workspace', 'TV', 'Family_kid_friendly', 'Host_greets_you', 'Smoking_allowed']
train = train[feature_columns]

train.head()
train = train.drop('bathrooms', axis=1)

train = train.drop('security_deposit', axis=1)

train = train.drop('extra_people', axis=1)

train = train.drop('distance', axis=1)

train = train.drop('bed_type', axis=1)

train = train.drop('minimum_nights', axis=1)

train = train.drop('cancellation_policy', axis=1)

train = train.drop('Laptop_friendly_workspace', axis=1)

train = train.drop('Family_kid_friendly', axis=1)

train = train.drop('Host_greets_you', axis=1)

train = train.drop('Smoking_allowed', axis=1)

train = train.drop('TV', axis=1)
kek = kek.drop('TV', axis=1)

kek = kek.drop('bathrooms', axis=1)

kek = kek.drop('security_deposit', axis=1)

kek = kek.drop('extra_people', axis=1)

kek = kek.drop('distance', axis=1)

kek = kek.drop('bed_type', axis=1)

kek = kek.drop('minimum_nights', axis=1)

kek = kek.drop('cancellation_policy', axis=1)

kek = kek.drop('Laptop_friendly_workspace', axis=1)

kek = kek.drop('Family_kid_friendly', axis=1)

kek = kek.drop('Host_greets_you', axis=1)

kek = kek.drop('Smoking_allowed', axis=1)
train.corr().style.format("{:.2}").background_gradient(cmap='coolwarm', axis=1)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
Y = train[tar].values

X = train.drop(tar, axis=1)

X.shape, Y.shape
# Разбиваем данные на обучающую и тестовую часть.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)



# Создание модели, которая будет подбирать веса для признаков.

model = LinearRegression()



# Просим модель подобрать веса для признаков.

model.fit(X_train, Y_train)



# Предсказываем значения с помощью модели.

pred_train = model.predict(X_train)

pred_test = model.predict(X_test)
mean_squared_error(Y_train, pred_train) ** 0.5
train = train[train.dtypes[(train.dtypes != object)].index]

kek = kek[kek.dtypes[(kek.dtypes != object)].index]
model.fit(train.drop('price', axis=1), train['price'])
predictions = model.predict(kek)
sam.head()
sam['price'] = predictions

sam.to_csv('my_submission.csv', index=False)