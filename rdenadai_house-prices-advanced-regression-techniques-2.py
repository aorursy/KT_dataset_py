import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



sns.set(style="white")

%matplotlib inline



def rmse(y_test, y_pred):

      return np.sqrt(mean_squared_error(y_test, y_pred))



def rmsle(y_test, y_pred):

    return np.sqrt(mean_squared_log_error(y_test, y_pred))
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_train.head(10)
df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)



size = len(list(df_train.columns))-1



# df_train.dropna(axis=1, inplace=True)

# f_columns = list(df_train.columns)

# size = len(f_columns)-1

# del f_columns[size]

# df_test = df_test[f_columns]

# df_train.head()
def preprocessing(df):

    simputer = SimpleImputer(strategy="most_frequent")

    

    df = df.replace(np.nan, 0).replace(np.inf, 1e+5).replace(-np.inf, -1e+5)

    for column in df.columns:

        if df[column].dtype.name == "object":

            df[column] = pd.Categorical(df[column]).codes



    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    num_cols = list(df.select_dtypes(include=numerics).columns)

    df[num_cols] = simputer.fit_transform(df[num_cols])

    return df



def normalization(df, norm):

    columns = df.columns

    return pd.DataFrame(norm.transform(df), columns=columns)



columns = df_train.columns

x_train = pd.DataFrame(df_train.to_numpy()[:, :size], columns=columns[:size])

y_train = df_train.to_numpy()[:, size:].ravel().astype(np.float64)



df_train = preprocessing(x_train).astype(np.int)

df_test = preprocessing(df_test).astype(np.int)



# norm = StandardScaler().fit(df_train)

# df_train = normalization(df_train, norm).astype(np.float64)

# df_test = normalization(df_test, norm).astype(np.float64)



# norm = MinMaxScaler().fit(df_train)

# df_train = normalization(df_train, norm).astype(np.float64)

# df_test = normalization(df_test, norm).astype(np.float64)



print(df_train.shape)

print(df_test.shape)

print(y_train.shape)
df_train.head(10)
concat = np.c_[df_train.to_numpy(), y_train.reshape(len(y_train), 1)]

df = pd.DataFrame(concat, columns=list(df_train.columns) + ["Label"]).astype(np.float64)

corr = df.corr()

cmap = sns.diverging_palette(10, 255, as_cmap=True)



plt.figure(figsize=(45, 45))

plt.subplot(2, 1, 1)

plt.title("Pearson Correlation")

ax = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=1, annot=False, cbar_kws={"shrink": .5})

ax.set_ylim(size, 0)

ax.set_xlim(0, size)

plt.tight_layout()

plt.show()
columns = df_train.columns

columns = columns[tuple(np.where(columns != "Label"))]

pca = PCA(n_components=len(columns))

pca.fit(df_train[columns])



print("PCA (Principal component analysis):")

print("Singular Values:")

print(np.round(pca.singular_values_, 5))
columns = list(df.columns[corr["Label"].ravel() > 0.05].ravel())

columns += list(df.columns[corr["Label"].ravel() < -0.05].ravel())

columns = np.array(sorted(columns))

columns = columns[tuple(np.where(columns != "Label"))]

print("Columns with strong linear correlation:")

print(len(columns))

print(columns)
pca = PCA(n_components=len(columns))

pca.fit(df_train[columns])



print("PCA (Principal component analysis):")

print("Singular Values:")

print(np.round(pca.singular_values_, 5))
pX_train, pX_test, py_train, py_test = train_test_split(df_train[columns], y_train, test_size=0.33, random_state=42)
print("Random Forest Regression:")

md = RandomForestRegressor(**{

    'criterion': 'mse',

    'max_features': 'auto',

    'max_leaf_nodes': 500,

    'n_estimators': 500,

    'n_jobs': 4,

    'random_state': 42

})

md.fit(pX_train, py_train)

y_pred = md.predict(pX_test)

print(f"R^2: {md.score(pX_test, py_test)}")

print(f"RMSE: {rmse(py_test, y_pred)}")

print(f"RMSLE: {rmsle(py_test, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(py_test, y_pred)}")

print(f"MAE: {mean_absolute_error(py_test, y_pred)}")
# print("MLP Regression:")

# md = MLPRegressor(**{

#     "hidden_layer_sizes": (100, 10),

#     "learning_rate": "constant",

#     "learning_rate_init": 1e-3,

#     "activation": "relu",

#     "alpha": 1e-2,

#     "batch_size": 32,

#     "solver": "lbfgs",

#     "max_iter": 500,

#     "random_state": 42

# })

# md.fit(pX_train, py_train)

# y_pred = md.predict(pX_test)

# print(f"R^2: {md.score(pX_test, py_test)}")

# print(f"RMSE: {rmse(py_test, y_pred)}")

# print(f"RMSLE: {rmsle(py_test, y_pred)}")

# print(f"Log MSE: {mean_squared_log_error(py_test, y_pred)}")

# print(f"MAE: {mean_absolute_error(py_test, y_pred)}")
pca = PCA(n_components=30)

x_pca = pca.fit_transform(df_train[columns])

print("PCA (Principal component analysis):")

print("Singular Values:")

print(np.round(pca.singular_values_, 5))



pX_train, pX_test, py_train, py_test = train_test_split(x_pca, y_train, test_size=0.33, random_state=42)
print("Random Forest Regression:")

md = RandomForestRegressor(**{

    'criterion': 'mse',

    'max_features': 'auto',

    'max_leaf_nodes': 500,

    'n_estimators': 500,

    'n_jobs': 4,

    'random_state': 42

})

md.fit(pX_train, py_train)

y_pred = md.predict(pX_test)

print(f"R^2: {md.score(pX_test, py_test)}")

print(f"RMSE: {rmse(py_test, y_pred)}")

print(f"RMSLE: {rmsle(py_test, y_pred)}")

print(f"Log MSE: {mean_squared_log_error(py_test, y_pred)}")

print(f"MAE: {mean_absolute_error(py_test, y_pred)}")
# print("MLP Regression:")

# md = MLPRegressor(**{

#     "hidden_layer_sizes": (100, 10),

#     "learning_rate": "constant",

#     "learning_rate_init": 1e-3,

#     "activation": "relu",

#     "alpha": 1e-4,

#     "batch_size": 32,

#     "solver": "lbfgs",

#     "max_iter": 500,

#     "random_state": 42

# })

# md.fit(pX_train, py_train)

# y_pred = md.predict(pX_test)

# print(f"R^2: {md.score(pX_test, py_test)}")

# print(f"RMSE: {rmse(py_test, y_pred)}")

# print(f"RMSLE: {rmsle(py_test, y_pred)}")

# print(f"Log MSE: {mean_squared_log_error(py_test, y_pred)}")

# print(f"MAE: {mean_absolute_error(py_test, y_pred)}")
md = RandomForestRegressor(**{

    'criterion': 'mse',

    'max_features': 'auto',

    'max_leaf_nodes': 500,

    'n_estimators': 2500,

    'n_jobs': 4,

    'random_state': 42

})

md.fit(df_train[columns], y_train)

y_pred = np.round(md.predict(df_test[columns]).ravel(), 3)



# md = MLPRegressor(**{

#     "hidden_layer_sizes": (200, 5),

#     "learning_rate": "adaptive",

#     "learning_rate_init": 1e-1,

#     "activation": "relu",

#     "alpha": 1e-5,

#     "batch_size": 32,

#     "solver": "lbfgs",

#     "max_iter": 2000,

#     "random_state": 42

# })

# md.fit(x_pca, y_train)

# y_pred = np.round(md.predict(t_pca).ravel(), 3)



i_df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sub = pd.DataFrame()

sub["Id"] = i_df_test["Id"]

sub["SalePrice"] = y_pred.astype(np.int)

sub.to_csv('submission.csv', index=False)