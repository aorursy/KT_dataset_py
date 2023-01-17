# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/housing.csv')

data.info()
data.head()
y = data.median_house_value
data_features = ['housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
x = data[data_features]
x.describe()
x.head()
from sklearn.tree import DecisionTreeRegressor

data_model = DecisionTreeRegressor(random_state=1)
data_model
data_model.fit(x,y)
print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(data_model.predict(x.head()))
