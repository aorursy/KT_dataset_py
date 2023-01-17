# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
training = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
training.head()
import matplotlib.pyplot as plt

price_hist = plt.hist(training['SalePrice'], bins=40)
import scipy.stats as scipy

sales_mean = training['SalePrice'].mean()

sales_var = training['SalePrice'].var()

sales_skew = scipy.skew(training['SalePrice'])

sales_kurt = scipy.kurtosis(training['SalePrice'])



print("The first four moments of SalePrice are")

print("------------")

print("1. Mean:") 

print(sales_mean)

print("------------")

print("2. Variance:") 

print(sales_var)

print("------------")

print("3. Skewness:") 

print(sales_skew)

print("------------")

print("4. Kurtosis:") 

print(sales_kurt)
import seaborn as sns

corr = training.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr, vmax=.8, square=True)
selected_features = ['OverallQual','YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageArea']

training[selected_features].isnull().sum()
import xgboost

from sklearn import model_selection

from sklearn.metrics import accuracy_score
train_X = training[selected_features]

train_y = training['SalePrice']

test_X = test[selected_features]
xgb_model = xgboost.XGBRegressor()

xgb_model.fit(train_X, train_y)
prediction = xgb_model.predict(test_X)

print(prediction)

print(len(prediction))
np.savetxt('house_prices_prediction.csv',prediction, delimiter=',')