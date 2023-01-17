import pandas as pd

data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data.head()
data.shape
data.isnull().values.any()
data[data.isnull().T.any().T]
data.head()
data.hist(bins = 10, figsize=(18, 16), color="#2c5af2")
data.describe()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() 

corr_matrix = data.corr()

corr_matrix



plt.figure(figsize=(25,25))

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap="YlGnBu")
data.info()
data.isnull().sum()
from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
y=data['SalePrice'].values

y=y.reshape(-1,1)

X=data[['OverallQual','TotalBsmtSF','1stFlrSF','GarageArea','GarageArea','GrLivArea']].values

XX=data[['OverallQual','TotalBsmtSF','1stFlrSF','GarageArea','GarageArea','GrLivArea']]
XX=data[['OverallQual','TotalBsmtSF','1stFlrSF','GarageArea','GarageArea','GrLivArea']]

XX.isnull().sum()
XX.info()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
lm =  RandomForestClassifier()

lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
from matplotlib import pyplot as plt

plt.scatter(y_test, predictions)

plt.xlabel("Actual Income")

plt.ylabel("Predicted Income")
clf.score(X_test, y_test)
from yellowbrick.regressor import PredictionError

from sklearn.linear_model import Lasso

lasso = Lasso()

visualizer = PredictionError(lasso)

visualizer.fit(X_train, y_train)

visualizer.score(X_test, y_test)

g = visualizer.poof()
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import metrics

scores = cross_val_score(clf, X, y, cv=10)

scores
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold

from yellowbrick.model_selection import CVScores



# Create a new figure and axes

_, ax = plt.subplots()

cv = KFold(10)



oz = CVScores(linear_model.LinearRegression(), ax=ax, cv=cv, scoring='r2')



oz.fit(X, y)

oz.poof()