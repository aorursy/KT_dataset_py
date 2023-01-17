# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/automobile-dataset/Automobile_data.csv")
dataset.info()
dataset['price'] = dataset['price'].str.replace('?','0').astype('float')

dataset["normalized-losses"] = dataset["normalized-losses"].str.replace('?','0').astype('float')
dataset["num-of-doors"] = dataset["num-of-doors"].str.replace('?','0').astype('float')
dataset["bore"] = dataset["bore"].str.replace('?','0').astype('float')
dataset["stroke"] = dataset["stroke"].str.replace('?','0').astype('float')
dataset["horsepower"] = dataset["horsepower"].str.replace('?','0').astype('float')
dataset["peak-rpm"] = dataset["peak-rpm"].str.replace('?','0').astype('float')
dataset.hist(bins=50,figsize=(20,15))
plt.show()
correlation = dataset.corr()
correlation

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
X = ['wheel-base', 'length', 'width', 'curb-weight', 'engine-size', 'horsepower', 'city-mpg', 'highway-mpg']
train_X = train_set[X]
train_Y = train_set['price']
test_X = test_set[X]
test_Y = test_set['price']
train_X.hist(bins=50,figsize=(20,15))
plt.show
train_Y.hist(bins=50,figsize=(10,5))
plt.show()
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
fuel_enc = train_set["fuel-type"]
fuel_enc1 = encoder.fit_transform(fuel_enc)

print(fuel_enc1)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
aspiration_enc = train_set["aspiration"]
aspiration_enc1=encoder.fit_transform(aspiration_enc)
aspiration_enc1
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
num_door_enc = train_set["num-of-doors"]
num_door_enc1=encoder.fit_transform(num_door_enc)
num_door_enc1

#selecting a training model

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_X,train_Y)
predict = lin_reg.predict(test_X[:5])
print(predict)
from sklearn.metrics import mean_squared_error
predict = lin_reg.predict(train_X)
lin_mse = mean_squared_error(train_Y,predict)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
lin_reg.score(test_X,test_Y)
from sklearn.ensemble import RandomForestRegressor
randomF = RandomForestRegressor()
randomF.fit(train_X,train_Y)
predictF = randomF.predict(test_X[:5])
print(predictF)

randomF.score(test_X,test_Y)
randomF.score(train_X,train_Y)
