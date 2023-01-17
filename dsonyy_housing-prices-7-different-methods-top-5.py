import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import display

pd.options.display.max_columns = None
train = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col=0)

X_test = pd.read_csv("../input/home-data-for-ml-course/test.csv", index_col=0)
X = train.iloc[:, :-1]

y = train.iloc[:, -1:]
cols_with_missing = X.isnull().any()

cols_with_missing = cols_with_missing[cols_with_missing].index



cols_missing_factor = pd.Series(X[cols_with_missing].isnull().sum() / len(X) * 100,

                                index=cols_with_missing)



MISSING_VALUES_THRESHOLD = 50
values = ((col, round(perc, 1), perc >= MISSING_VALUES_THRESHOLD) for col, perc in cols_missing_factor.iteritems())

columns = ["Columns with missing values", "percentage of missing values", "to drop"]



pd.DataFrame(values, columns=columns)
cols_to_drop = cols_missing_factor[cols_missing_factor >= MISSING_VALUES_THRESHOLD].index.to_list()

X = X.drop(columns=cols_to_drop)

X_test = X_test.drop(columns=cols_to_drop)
X.columns = range(X.shape[1])

X_test.columns = range(X.shape[1])



num_cols = X.select_dtypes(include=np.number).columns.to_list()

cat_cols = X.select_dtypes(exclude=np.number).columns.to_list()
from sklearn.impute import SimpleImputer



num_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

X[num_cols] = num_imputer.fit_transform(X[num_cols])

X_test[num_cols] = num_imputer.transform(X_test[num_cols])



cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
X = X.values

y = y.values

X_test = X_test.values
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=0)
from sklearn.preprocessing import StandardScaler



X_std_scaler = StandardScaler()

X_train[:, num_cols] = X_std_scaler.fit_transform(X_train[:, num_cols])

X_val[:, num_cols] = X_std_scaler.transform(X_val[:, num_cols])



y_std_scaler = StandardScaler()

y_train = y_std_scaler.fit_transform(y_train)

y_val = y_std_scaler.transform(y_val)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



col_trans = ColumnTransformer([("encoder", OneHotEncoder(), cat_cols)], remainder="passthrough")

col_trans.fit(X)

X_train = col_trans.transform(X_train).toarray()

X_val = col_trans.transform(X_val).toarray()
pd.DataFrame(X_train)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error



lin_reg = LinearRegression().fit(X_train, y_train)



y_pred = lin_reg.predict(X_val)

lin_reg_score = mean_absolute_error(y_val, y_pred)

print("MAE:", lin_reg_score)
from sklearn.preprocessing import PolynomialFeatures



X_train_poly = X_train[:, num_cols]

X_train_poly = PolynomialFeatures(degree=2).fit_transform(X_train)

X_train_poly = np.concatenate((X_train_poly, np.delete(X_train_poly, num_cols, axis=1)), axis=1)



X_val_poly = X_val[:, num_cols]

X_val_poly = PolynomialFeatures(degree=2).fit_transform(X_val)

X_val_poly = np.concatenate((X_val_poly, np.delete(X_val_poly, num_cols, axis=1)), axis=1)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error



poly_reg = LinearRegression().fit(X_train_poly, y_train)



y_pred = poly_reg.predict(X_val_poly)

poly_reg_score = mean_absolute_error(y_val, y_pred)

print("MAE:", poly_reg_score)
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error



svr = SVR(kernel="rbf").fit(X_train, y_train)



y_pred = svr.predict(X_val)

svr_score = mean_absolute_error(y_val, y_pred)

print("MAE:", svr_score)
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error



tree = DecisionTreeRegressor().fit(X_train, y_train)



y_pred = tree.predict(X_val)

tree_score = mean_absolute_error(y_val, y_pred)

print("MAE:", tree_score)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest = RandomForestRegressor().fit(X_train, y_train)



y_pred = forest.predict(X_val)

forest_score = mean_absolute_error(y_val, y_pred)

print("MAE:", forest_score)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



xgb = XGBRegressor().fit(X_train, y_train)



y_pred = xgb.predict(X_val)

xgb_score = mean_absolute_error(y_val, y_pred)

print("MAE:", xgb_score)
from keras import models, layers

from sklearn.metrics import mean_absolute_error



net = models.Sequential()

net.add(layers.Dense(128, input_shape=(X_train.shape[1],), activation="relu"))

net.add(layers.Dropout(0.2))

net.add(layers.Dense(64, activation="relu"))

net.add(layers.Dense(1, activation=None))



net.compile(optimizer="rmsprop", loss="mae", metrics=["mae"])



history = net.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=100, verbose=0)



y_pred = net.predict(X_val)

nn_score = mean_absolute_error(y_val, y_pred)

print("MAE:", nn_score)