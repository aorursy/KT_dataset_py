# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras import Sequential
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.info()
data.columns.values
import seaborn as sns
sns.pairplot(data, hue="Species")
data.info()
if 'Species' in data.columns.values:
    data_y = data["Species"]
    data.drop(["Species", "Id"], axis = 1, inplace=True)
data.info()
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data.values)

scaled_data = pd.DataFrame(x_scaled)
scaled_data.head()
scaled_data.info()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data_y = label_encoder.fit_transform(data_y)

data_y
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_model.fit(scaled_data, data_y)

rf_model.score(scaled_data, data_y)
pred_y = rf_model.predict(scaled_data).round().astype(int)
pred_y
