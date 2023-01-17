# Loading stuff

import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



sns.set()

pd.set_option('max_columns', 1000)

warnings.filterwarnings('ignore')

%matplotlib inline
# Load the data

df_train = pd.read_csv('../input/train.csv')
# Explore the columns

print(df_train.columns.values)

print('No. variables:', len(df_train.columns.values))
df_train.describe()
num_missing = df_train.isnull().sum()

percent = num_missing / df_train.isnull().count()



df_missing = pd.concat([num_missing, percent], axis=1, keys=['MissingValues', 'Fraction'])

df_missing = df_missing.sort_values('Fraction', ascending=False)

df_missing[df_missing['MissingValues'] > 0]
variables_to_keep = df_missing[df_missing['MissingValues'] == 0].index

df_train = df_train[variables_to_keep]
# Build the correlation matrix

matrix = df_train.corr()

f, ax = plt.subplots(figsize=(16, 12))

sns.heatmap(matrix, vmax=0.7, square=True)
interesting_variables = matrix['SalePrice'].sort_values(ascending=False)

# Filter out the target variables (SalePrice) and variables with a low correlation score (v such that -0.6 <= v <= 0.6)

interesting_variables = interesting_variables[abs(interesting_variables) >= 0.6]

interesting_variables = interesting_variables[interesting_variables.index != 'SalePrice']

interesting_variables
values = np.sort(df_train['OverallQual'].unique())

print('Unique values of "OverallQual":', values)
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)

data.plot.scatter(x='OverallQual', y='SalePrice')
cols = interesting_variables.index.values.tolist() + ['SalePrice']

sns.pairplot(df_train[cols], size=2.5)

plt.show()
# Build the correlation matrix

matrix = df_train[cols].corr()

f, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(matrix, vmax=1.0, square=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



pred_vars = [v for v in interesting_variables.index.values if v != 'SalePrice']

target_var = 'SalePrice'



X = df_train[pred_vars]

y = df_train[target_var]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# Build a plot

plt.scatter(y_pred, y_test)

plt.xlabel('Prediction')

plt.ylabel('Real value')



# Now add the perfect prediction line

diagonal = np.linspace(0, np.max(y_test), 100)

plt.plot(diagonal, diagonal, '-r')

plt.show()
from sklearn.metrics import mean_squared_log_error, mean_absolute_error



print('MAE:\t$%.2f' % mean_absolute_error(y_test, y_pred))

print('MSLE:\t%.5f' % mean_squared_log_error(y_test, y_pred))