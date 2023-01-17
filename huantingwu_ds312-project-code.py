import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

np.random.seed(42)
df_raw = pd.read_csv('/kaggle/input/statistics-homework/MPG.csv')

df_full = df_raw.copy()
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()

origin = df_full['origin'].to_numpy().reshape(-1, 1)

origin_encoded_array = onehot_encoder.fit_transform(origin).toarray()

assert df_full.shape[0] == origin_encoded_array.shape[0]



origin_encoded_df = pd.DataFrame(origin_encoded_array, columns=['origin_A', 'origin_B', 'origin_C']).astype('int')

df_encoded_origin = pd.concat([df_full, origin_encoded_df], axis=1).drop('origin', axis=1)

assert df_full.shape[1] + origin_encoded_df.shape[1] == pd.concat([df_full, origin_encoded_df], axis=1).shape[1]



categorical_columns = ['cylinders', 'modelyear']

numerical_columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']

assert (len(categorical_columns) + len(numerical_columns) + origin_encoded_df.shape[1]) == df_encoded_origin.shape[1]



df_encoded_ordinal = df_encoded_origin.copy()



from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ordinal_encoder = OrdinalEncoder()



for categorical_column in categorical_columns:

    df_encoded_ordinal[categorical_column] = ordinal_encoder.fit_transform(df_encoded_ordinal[categorical_column].to_numpy().reshape(-1, 1)).astype('int')



df_scaled_numerical = df_encoded_ordinal.copy()



from sklearn.preprocessing import StandardScaler

normal_scaler = StandardScaler()



for numerical_column in numerical_columns:

    df_scaled_numerical[numerical_column] = normal_scaler.fit_transform(df_scaled_numerical[numerical_column].to_numpy().reshape(-1, 1))



df_preprocessed = df_scaled_numerical.copy()



train, test = df_preprocessed.iloc[:360], df_preprocessed.iloc[360:]

train_X, train_y_truth = train[train.columns[1:]], train[train.columns[0]]

test_X, test_y_truth = test[test.columns[1:]], test[test.columns[0]]
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_model = linear_regressor.fit(train_X, train_y_truth)

train_y_pred_linear = linear_model.predict(train_X)

linear_residuals = train_y_pred_linear - train_y_truth
plt.figure(figsize=(15,7), dpi=300)

sns.residplot(train_y_pred_linear, train_y_truth)
from statsmodels.graphics.gofplots import qqplot

fig, ax = plt.subplots(figsize=(15, 7), dpi=300)

qqplot(linear_residuals, line='45', ax=ax);
train_r2_linear = linear_model.score(train_X, train_y_truth)

test_r2_linear = linear_model.score(test_X, test_y_truth)

test_y_pred_linear = linear_model.predict(test_X)

from sklearn.metrics import mean_squared_error

linear_mse = mean_squared_error(test_y_truth, test_y_pred_linear)
print('train_r2_linear:', train_r2_linear)

print('test_r2_linear:', test_r2_linear)

print('linear prediction mean squared error:', linear_mse)
for variable, coefficient in zip(train_X.columns, linear_model.coef_):

    print(variable + ': '+ str(coefficient))

print('intercept:', linear_model.intercept_)
from sklearn.ensemble import RandomForestRegressor

random_forest_regressor = RandomForestRegressor()

random_forest_model = random_forest_regressor.fit(train_X, train_y_truth)

train_r2_random_forest = random_forest_model.score(train_X, train_y_truth)

test_r2_random_forest = random_forest_model.score(test_X, test_y_truth)
test_y_pred_random_forest = random_forest_model.predict(test_X)

random_forest_mse = mean_squared_error(test_y_truth, test_y_pred_random_forest)
print('train_r2_random_forest:', train_r2_random_forest)

print('test_r2_random_forest:', test_r2_random_forest)

print('random forest prediction mean squared error:', random_forest_mse)