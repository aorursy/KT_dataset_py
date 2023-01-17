import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import LinearSVR, SVR

from sklearn.model_selection import GridSearchCV



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



sns.set(style="white")

%matplotlib inline
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_train.head(10)
df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)
def preprocessing(df):

    for column in df.columns:

        df[column].fillna(0)

        if df[column].dtype.name == "object":

            df[column] = pd.Categorical(df[column]).codes

    return df



def normalization(df, norm):

    columns = df.columns

    return pd.DataFrame(norm.transform(df), columns=columns)



columns = df_train.columns

x_train = pd.DataFrame(df_train.to_numpy()[:, :79], columns=columns[:79])

y_train = df_train.to_numpy()[:, 79:].ravel().astype(np.float64)



df_train = preprocessing(x_train)

df_test = preprocessing(df_test)

norm = StandardScaler().fit(df_train).fit(df_test)

df_train = normalization(df_train, norm).astype(np.float64)

df_test = normalization(df_test, norm).astype(np.float64)



print(df_train.shape)

print(df_test.shape)
df_train.head(10)
df = pd.DataFrame(np.c_[df_train.to_numpy(), y_train.reshape(len(y_train), 1)], columns=list(df_train.columns) + ["Label"]).astype(np.float64)

corr = df.corr()

cmap = sns.diverging_palette(10, 255, as_cmap=True)



plt.figure(figsize=(45, 45))

plt.subplot(2, 1, 1)

plt.title("Pearson Correlation")

ax = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=1, annot=False, cbar_kws={"shrink": .5})

ax.set_ylim(81, 0)

ax.set_xlim(0, 81)

plt.tight_layout()

plt.show()
columns = list(df.columns[corr["Label"].ravel() > 0.25].ravel())

columns += list(df.columns[corr["Label"].ravel() < -0.25].ravel())

columns = np.array(sorted(columns))

columns = columns[tuple(np.where(columns != "Label"))]

print("Columns with strong linear correlation:")

print(columns)
def rmse(y_test, y_pred):

      return np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression:")

md = LinearRegression(n_jobs=4)

md.fit(df_train[columns], y_train)

y_pred = md.predict(df_train[columns])

print(f"R^2: {md.score(df_train[columns], y_train)}")

print(f"RMSE: {rmse(y_train, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(y_train, y_pred)}")

print(f"MAE: {mean_absolute_error(y_train, y_pred)}")
print("Ridge Regression:")



params = {

    "alpha": np.linspace(1e-4, 5, 80),

    "solver" : ["svd", "cholesky", "lsqr", "sparse_cg"]

}

clf = GridSearchCV(Ridge(), params, cv=3, n_jobs=8, verbose=5)

clf.fit(df_train[columns], y_train)

print("Best params: ", clf.best_params_)



md = Ridge(**clf.best_params_)

md.fit(df_train[columns], y_train)

y_pred = md.predict(df_train[columns])

print(f"R^2: {md.score(df_train[columns], y_train)}")

print(f"RMSE: {rmse(y_train, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(y_train, y_pred)}")

print(f"MAE: {mean_absolute_error(y_train, y_pred)}")
print("Random Forest Regression:")



params = {

    "n_estimators": [15, 25, 35, 45, 55],

    "criterion" : ["mse", "mae"],

    "min_samples_split": [2, 4, 6, 8],

    "max_features": ["sqrt", "log2", "auto"]

}

clf = GridSearchCV(RandomForestRegressor(), params, cv=3, n_jobs=8, verbose=5)

clf.fit(df_train[columns], y_train)

print("Best params: ", clf.best_params_)



md = RandomForestRegressor(**clf.best_params_)

md.fit(df_train[columns], y_train)

y_pred = md.predict(df_train[columns])

print(f"R^2: {md.score(df_train[columns], y_train)}")

print(f"RMSE: {rmse(y_train, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(y_train, y_pred)}")

print(f"MAE: {mean_absolute_error(y_train, y_pred)}")
print("Linear SVM Regression:")



params = {

    "C": np.linspace(200, 220, 20),

    "loss": ["epsilon_insensitive"],

    "max_iter": [9000]

}

clf = GridSearchCV(LinearSVR(), params, cv=3, n_jobs=8, verbose=5)

clf.fit(df_train[columns], y_train)

print("Best params: ", clf.best_params_)



md = LinearSVR(**clf.best_params_)

md.fit(df_train[columns], y_train)

y_pred = md.predict(df_train[columns])

print(f"R^2: {md.score(df_train[columns], y_train)}")

print(f"RMSE: {rmse(y_train, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(y_train, y_pred)}")

print(f"MAE: {mean_absolute_error(y_train, y_pred)}")
print("SVM Regression:")



params = {

    "kernel": ["rbf", "poly", "sigmoid"],

    "degree": [3, 5, 6, 9],

    "gamma": ["scale"],

    "C": [1, 2, 3, 4, 5, 6, 10, 20, 50]

}

clf = GridSearchCV(SVR(), params, cv=3, n_jobs=8, verbose=5)

clf.fit(df_train[columns], y_train)

print("Best params: ", clf.best_params_)



md = SVR(**clf.best_params_)

md.fit(df_train[columns], y_train)

y_pred = md.predict(df_train[columns])

print(f"R^2: {md.score(df_train[columns], y_train)}")

print(f"RMSE: {rmse(y_train, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(y_train, y_pred)}")

print(f"MAE: {mean_absolute_error(y_train, y_pred)}")
print("Multi Layer Perceptron Regression:")



params = {

    "hidden_layer_sizes": [(100, ), (100, 2), (25, 10), (50, 7)],

    "learning_rate": ["constant", "invscaling", "adaptive"],

    "learning_rate_init": [1e-1, 1e-2, 1e-3, 1e-4],

    "solver": ["lbfgs", "adam"]

}

clf = GridSearchCV(MLPRegressor(), params, cv=3, n_jobs=8, verbose=5)

clf.fit(df_train, y_train)

print("Best params: ", clf.best_params_)



md = MLPRegressor(**clf.best_params_)

md.fit(df_train[columns], y_train)

y_pred = md.predict(df_train[columns])

print(f"R^2: {md.score(df_train[columns], y_train)}")

print(f"RMSE: {rmse(y_train, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(y_train, y_pred)}")

print(f"MAE: {mean_absolute_error(y_train, y_pred)}")
md = RandomForestRegressor(**{'criterion': 'mae', 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 45})

md.fit(df_train[columns], y_train)

df_test = df_test.replace(np.nan, 0).replace(np.inf, 1e+10).replace(-np.inf, -1e+10)

y_pred = md.predict(df_test[columns]).ravel()

i_df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sub = pd.DataFrame()

sub["Id"] = i_df_test["Id"]

sub["SalePrice"] = y_pred

sub.to_csv('submission.csv', index=False)