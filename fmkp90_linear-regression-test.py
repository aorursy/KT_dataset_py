# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import scipy

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

test_data_id = test_data.id

print(train_data.shape, test_data.shape)
train_data = train_data[["full_sq","life_sq","floor","max_floor","build_year","num_room","price_doc"]]

train_data = train_data.dropna()

train_data_x = train_data.drop(["price_doc"],axis=1)

train_data_y = train_data.price_doc
test_data = test_data[["full_sq","life_sq","floor","max_floor","build_year","num_room"]]

test_data.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)

model.fit(train_data_x, train_data_y)
test_data.fillna(test_data.mean(), inplace=True);

model.predict(test_data)
test_data_y = model.predict(test_data)

model_predict_train = model.predict(train_data_x)

test_data.shape
#plt.scatter(test_data.full_sq[1:50],test_data_y[1:50])

plt.scatter(train_data_x.full_sq[1:500],model_predict_train[1:500])

plt.scatter(train_data_x.full_sq[1:500],train_data_y[1:500])
y = pd.DataFrame(test_data_y,columns=["price_doc"],index=test_data_id)
y[y<0] = 0

y.info()

y.hist()