import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn import metrics

import seaborn as sns

plt.style.use("bmh")
# Load data

boston_data = pd.read_csv("../input/boston/Boston.csv")
boston_data.head(15)
boston_data.shape
boston_data.columns
# check missing data

boston_data.isnull().sum()
sns.pairplot(boston_data, x_vars=['crim', 'zn', 'indus'], y_vars=["medv"])
sns.pairplot(boston_data, x_vars=['chas', 'nox', 'rm'], y_vars=["medv"])
sns.pairplot(boston_data, x_vars=['age', 'dis', 'rad'], y_vars=["medv"])
sns.pairplot(boston_data, x_vars=['tax', 'ptratio', 'black', 'lstat'], y_vars=["medv"])
corrMatrix = boston_data.corr()

fig, ax = plt.subplots(figsize=(18, 10))

sns.heatmap(corrMatrix, annot=True)
boston_data.dtypes
boston_data.describe()
# Now I will make all variables range from 0 to 1, because machine learning algorithms work better that way.



for column in boston_data.columns:

    maxcolumn = boston_data[column].max()

    if maxcolumn > 1:

        boston_data[column] = boston_data[column] / maxcolumn
boston_data.describe()
# Split data

boston_data_X = boston_data.drop("medv", axis=1)

boston_data_y = boston_data["medv"]
Xtrain, Xtest, ytrain, ytest = train_test_split(boston_data_X, boston_data_y, test_size=0.3, random_state=32)
linearparam = {"fit_intercept": [True, False], "normalize": [True, False], "copy_X": [True, False]}

lineargrid = GridSearchCV(LinearRegression(), linearparam, cv=10)

lineargrid.fit(Xtrain, ytrain)

print("Best Linear Regression estimator:", lineargrid.best_estimator_)
lassoparam = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 

              "fit_intercept": [True, False], "normalize": [True, False], "copy_X": [True, False]}

lassogrid = GridSearchCV(Lasso(), lassoparam, cv=10)

lassogrid.fit(Xtrain, ytrain)

print("Best Lasso Regression estimator:", lassogrid.best_estimator_)
ridgeparam = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 

              "fit_intercept": [True, False], "normalize": [True, False], "copy_X": [True, False]}

ridgegrid = GridSearchCV(Ridge(), ridgeparam, cv=10)

ridgegrid.fit(Xtrain, ytrain)

print("Best Ridge Regression estimator:", ridgegrid.best_estimator_)
ypredictedlinear = lineargrid.best_estimator_.predict(Xtest)



mae = metrics.mean_absolute_error(ytest, ypredictedlinear)

mse = metrics.mean_squared_error(ytest, ypredictedlinear)

r2 = metrics.r2_score(ytest, ypredictedlinear)



print("Linear Regression performance:")

print("MAE: {}".format(mae))

print("MSE: {}".format(mse))

print("R2 score: {}".format(r2))
ypredictedlasso = lassogrid.best_estimator_.predict(Xtest)



mae = metrics.mean_absolute_error(ytest, ypredictedlasso)

mse = metrics.mean_squared_error(ytest, ypredictedlasso)

r2 = metrics.r2_score(ytest, ypredictedlasso)



print("Lasso Regression performance:")

print("MAE: {}".format(mae))

print("MSE: {}".format(mse))

print("R2 score: {}".format(r2))
ypredictedridge = ridgegrid.best_estimator_.predict(Xtest)



mae = metrics.mean_absolute_error(ytest, ypredictedridge)

mse = metrics.mean_squared_error(ytest, ypredictedridge)

r2 = metrics.r2_score(ytest, ypredictedridge)



print("Ridge Regression performance:")

print("MAE: {}".format(mae))

print("MSE: {}".format(mse))

print("R2 score: {}".format(r2))