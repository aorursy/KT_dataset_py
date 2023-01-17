import numpy as np

import pandas as pd

from sklearn import datasets

import seaborn as sns

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
data = pd.read_csv('../input/kc_house_data.csv')

data.head(3)
print(data.shape)
print(data.isnull().any())
print(data.dtypes)
data['date'] = pd.to_datetime(data['date'])

data = data.set_index('id')

data.price = data.price.astype(int)

data.bathrooms = data.bathrooms.astype(int)

data.floors = data.floors.astype(int)

data.head(5)
data["house_age"] = data["date"].dt.year - data['yr_built']

data['renovated'] = data['yr_renovated'].apply(lambda yr: 0 if yr == 0 else 1)



data=data.drop('date', axis=1)

data=data.drop('yr_renovated', axis=1)

data=data.drop('yr_built', axis=1)

data.head(5)
pd.set_option('precision', 2)

print(data.describe())
correlation = data.corr(method='pearson')

columns = correlation.nlargest(10, 'price').index

columns
correlation_map = np.corrcoef(data[columns].values.T)

sns.set(font_scale=1.0)

heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)



plt.show()
data['price'] = np.log(data['price'])

data['sqft_living'] = np.log(data['sqft_living'])
X = data[columns]

Y = X['price'].values

X = X.drop('price', axis = 1).values
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=42)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))

pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))



results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=10, random_state=21)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
from sklearn.model_selection import GridSearchCV



scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

param_grid = dict(n_estimators=np.array([50,100,200,300,400]))

model = GradientBoostingRegressor(random_state=21)

kfold = KFold(n_splits=10, random_state=21)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)

grid_result = grid.fit(rescaledX, Y_train)



means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
from sklearn.metrics import mean_squared_error



scaler = StandardScaler().fit(X_train)

rescaled_X_train = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state=21, n_estimators=400)

model.fit(rescaled_X_train, Y_train)



# transform the validation dataset

rescaled_X_test = scaler.transform(X_test)

predictions = model.predict(rescaled_X_test)

print (mean_squared_error(Y_test, predictions))
compare = pd.DataFrame({'Prediction': predictions, 'Test Data' : Y_test})

compare.head(10)
actual_y_test = np.exp(Y_test)

actual_predicted = np.exp(predictions)

diff = abs(actual_y_test - actual_predicted)



compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})

compare_actual = compare_actual.astype(int)

compare_actual.head(5)