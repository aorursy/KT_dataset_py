import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv("../input/train.csv")

train.columns
print(train.head())
train['SalePrice'].describe()
sns.distplot(train['SalePrice']);
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_y = train.SalePrice

predictor_columns = ['OverallQual','GrLivArea','FullBath', 'TotRmsAbvGrd','YearBuilt', '1stFlrSF','YrSold']

#predictor_columns = ['OverallQual', 'GrLİvArea', 'GarageCars', 'GarageArea','TotalBsmtSF','1stFlrSF']
train_X = train[predictor_columns]

my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)
test = pd.read_csv("../input/test.csv")

print(test.head())
test_X = test[predictor_columns]
predicted_prices = my_model.predict(test_X)
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

print(my_submission)

my_submission.to_csv('submission.csv', index=False)