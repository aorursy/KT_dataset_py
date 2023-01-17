# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

#print(train_data.columns)
y = train_data['SalePrice']

features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtUnfSF','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces','GarageArea']

X = train_data[features]

#print(X)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.3)
def evaluate(prediction, val_y):

    score = 0

    for i in range(len(prediction)):

        distance = (prediction[i] - list(val_y)[i]) **2

        score += math.sqrt(distance)

    return score
#model testing

model = RandomForestRegressor(n_estimators = 70)

model.fit(train_X, train_y)

preds = model.predict(val_X)

evaluate(preds, val_y)
#testing process 

#RandomForestRegressor(max_depth = 50) -> 8854834

#RandomForestRegressor(max_depth = 61) -> 8766622

#max_depth = 61
test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

test = test_data[features].fillna(0)
regressor = RandomForestRegressor(n_estimators = 70, max_depth = 60)

regressor.fit(X, y)

prediction = regressor.predict(test)

#print(prediction)
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': prediction})

submission.to_csv('submission.csv', index = False)



pd.read_csv('submission.csv')