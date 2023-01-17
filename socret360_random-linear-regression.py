%matplotlib inline

import zipfile

import os

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

import numpy as np

from sklearn.metrics import mean_squared_error

from scipy import stats
PATH_DATASET = os.path.join("/kaggle/input/", "random-linear-regression")

TRAINING_DATASET = "train.csv"

TESTING_DATASET = "test.csv"
def load_dataset(filename, path_dataset=PATH_DATASET):

    print("LOADED: " + filename)

    return pd.read_csv(os.path.join(path_dataset, filename))

    

train = load_dataset(TRAINING_DATASET)

test = load_dataset(TESTING_DATASET)
data_points = train.copy()
data_points.head()
data_points.info()
data_points["y"].describe()
data_points.plot(kind="scatter", x="x", y="y")
corr_matrix = data_points.corr()
corr_matrix["y"]
data_points["y"].hist()
data_points.info()
data_points_prepared = data_points.copy()



data_points_prepared["y"].fillna(data_points_prepared["y"].mean(), inplace=True)
data_points_prepared.info()
data_points_prepared.plot(kind="scatter", x="x", y="y")
data_points_prepared = data_points.dropna()



data_points_prepared.info()
data_points_prepared.plot(kind="scatter", x="x", y="y")
data_points.plot(kind="scatter", x="x", y="y")
data_points_prepared_without_y = data_points_prepared.drop(['y'], axis=1)

data_points_prepared_y = data_points_prepared['y'].copy()
linear_regression = LinearRegression()

linear_regression.fit(data_points_prepared_without_y, data_points_prepared_y)
print (linear_regression.coef_, linear_regression.intercept_)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
lin_scores = cross_val_score(linear_regression, data_points_prepared_without_y, data_points_prepared_y, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
pd.Series(lin_rmse_scores).describe()
data_points_test_without_y = test.drop("y", axis=1);

data_points_test_y = test["y"].copy();



final_predictions = linear_regression.predict(data_points_test_without_y)



final_mse = mean_squared_error(data_points_test_y, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse
plt.title('Comparison of Y values in test and the Predicted values')

plt.ylabel('y')

plt.xlabel('x')

plt.scatter(data_points_test_without_y["x"], data_points_test_y, label="actual")

plt.scatter(data_points_test_without_y["x"], final_predictions, color="red", label="predicted")

plt.legend(loc='upper left')

plt.show()