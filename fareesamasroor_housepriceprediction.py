import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns

import pickle
housePriceData = pd.read_csv("../input/train_hp.csv")

housePriceData = housePriceData.dropna()
X = housePriceData.iloc[:, : 29]

y = housePriceData.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(housePriceData, y, test_size=0.2)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
sns.regplot(x=housePriceData['SalePrice'],y=housePriceData['TotalBsmtSF'])

plt.ylabel("SalePrice")

plt.xlabel("TotalBsmtSF")
sns.regplot(x=housePriceData['SalePrice'],y=housePriceData['YearBuilt'],ci=68)

plt.ylabel("SalePrice")

plt.xlabel("YearBuilt")
sns.regplot(x=housePriceData['SalePrice'],y=housePriceData['LotArea'])

plt.ylabel("SalePrice")

plt.xlabel("LotArea")
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regresso.fit(X, y)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[60,65,8450,7,5,2003,2003,706,0,150,856,856,854,1710,1,0,2,1,3,1,8,2003,2,548,0,0,0,2,2008]]))