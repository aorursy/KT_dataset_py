# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.columns
from sklearn.linear_model import LinearRegression



model = LinearRegression()



feature_column_names = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']



predicted_class_name = ['SalePrice']



X = train[feature_column_names].values

y = train[predicted_class_name].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



model.fit(X_train, y_train)
model.score(X_test, y_test)
new = test[feature_column_names].values



predicted_prices = model.predict(new)



predicted_price = np.reshape(predicted_prices, -1)



print(predicted_price)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_price})



submission.to_csv("houseprices.csv",index=False)