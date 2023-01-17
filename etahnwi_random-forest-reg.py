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
data = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")

data.head()
data.describe(include='all')
data.describe(include="all")
data.dtypes
data = data.rename(columns={"rent amount (R$)":"rent"})

#Cleaning Data

data['area'].fillna(data['area'].median(), inplace=True)

data['rooms'].fillna(data['rooms'].median(), inplace=True)

data['bathroom'].fillna(data['bathroom'].median(), inplace=True)

data['parking spaces'].fillna(data['parking spaces'].median(), inplace=True)

data.head()
# label encoding for city variables

from sklearn.preprocessing import LabelEncoder



# make copy to avoid changing original data

label_X

data.describe(include="all")
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
y = data.rent

variables = ['city','area','rooms','bathroom','parking spaces','furniture']

X = data[variables]



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=1)



# label encoding for city variables

from sklearn.preprocessing import LabelEncoder



s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



# make copy to avoid changing original data

label_X_train = X_train.copy()

label_X_test = X_test.copy()



#apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_test[col] = label_encoder.fit_transform(X_test[col])



print(label_X_train)
model = RandomForestRegressor()

house_cost =  model.fit(label_X_train,y_train)

house_cost.score(label_X_test,y_test)