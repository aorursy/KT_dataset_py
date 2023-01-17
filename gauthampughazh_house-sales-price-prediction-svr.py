import numpy as np

import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn_pandas import CategoricalImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from IPython.display import FileLink



pd.set_option('float_format', '{:.4f}'.format)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
corr_matrix = train_df.corr()

fig, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(corr_matrix, vmax=0.8, square=True)

plt.show()
# Selecting only the numeric columns

train = train_df.select_dtypes(['int64', 'float64'])

test = test_df.select_dtypes(['int64', 'float64'])



# Dropping the insignificant features

train = train.drop(['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars', 'Id'], axis=1)

test = test.drop(['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars', 'Id'], axis=1)
# Visualizing the unique value count in each feature

cat_count = train.apply(lambda x: x.value_counts().shape[0]).sort_values()

go.Figure(data=go.Bar(x=cat_count.index, y=cat_count))
# Categorical columns in numerical columns

cats_in_nums = cat_count.loc[cat_count < 50].index



# Converting the type of filtered numerical columns to categorical

train.loc[:, cats_in_nums] = train.loc[:, cats_in_nums].astype('object')

test.loc[:, cats_in_nums] = test.loc[:, cats_in_nums].astype('object')



train.info()
# Code block to view only n top correlated features



# top_features = corr_matrix.nlargest(15, 'SalePrice')['SalePrice'].index

# cm = np.corrcoef(corr_matrix.loc[top_features].values)

# sns.set(font_scale=1.25)

# fig = plt.figure(figsize=(12, 9))

# sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

#              annot_kws={'size': 10}, yticklabels=top_features.values,

#              xticklabels=top_features.values)

# plt.show()
mean_price = train_df.groupby('Neighborhood')['SalePrice'].mean().reset_index()

px.bar(mean_price, x='Neighborhood', y='SalePrice')
three_bath_houses = train_df[(train_df['FullBath'] == 3)]

three_bath_houses_neigh = three_bath_houses.groupby('Neighborhood')['SalePrice'].mean().reset_index()

px.bar(three_bath_houses_neigh, x='Neighborhood', y='SalePrice')
large_garage_houses = train_df[(train_df['GarageArea'] >= 800)]

large_garage_houses_neigh = large_garage_houses.groupby('Neighborhood')['SalePrice'].mean().reset_index()

px.bar(large_garage_houses_neigh, x='Neighborhood', y='SalePrice')
# Adding Neighborhood feature to the filtered datasets

train['Neighborhood'] = train_df['Neighborhood']

test['Neighborhood'] = test_df['Neighborhood']



# Extracting the SalePrice from training data

y_train = np.log(train['SalePrice'].values)

train = train.drop(['SalePrice'], axis=1)
train.info()
test.info()
px.histogram(train, 'GarageArea')
figure = plt.figure(figsize=(15, 9))

ax = sns.distplot(train['GarageArea'])
px.histogram(train, 'TotalBsmtSF')
figure = plt.figure(figsize=(15, 9))

ax = sns.distplot(train['TotalBsmtSF'])
figure = plt.figure(figsize=(15, 9))

ax = sns.distplot(train['MasVnrArea'].fillna(0))
imputer = SimpleImputer(strategy='median')

train_num = imputer.fit_transform(train.select_dtypes(['int64', 'float64']))

test_num = imputer.transform(test.select_dtypes(['int64', 'float64']))
cat_columns = train.select_dtypes(['object']).columns

cat_imputer = CategoricalImputer()

train_cat = cat_imputer.fit_transform(train.loc[:, cat_columns].values)

test_cat = cat_imputer.transform(test.loc[:, cat_columns].values)
one_hot_encoder = OneHotEncoder()

combined_cat_data = np.vstack([train_cat, test_cat])

combined_cat = one_hot_encoder.fit_transform(combined_cat_data).todense()



train_cat = combined_cat[:train.shape[0]]

test_cat = combined_cat[train.shape[0]:]
# Creating a scaler for input features

X_scaler = MinMaxScaler()



# Transforming the input features of both train and test

train_num = X_scaler.fit_transform(train_num)

test_num = X_scaler.transform(test_num)
X_train = np.hstack((train_num, train_cat))

X_test = np.hstack((test_num, test_cat))
models = [AdaBoostRegressor(learning_rate=2),

          SVR(kernel='linear'),

          RandomForestRegressor(n_estimators=200, random_state=1)]

errors = []

for model in models:

    model_name = model.__class__.__name__

    error = np.sqrt(abs(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))).mean()

    errors.append([model_name, error])

    

result_df = pd.DataFrame(errors, columns=['Model name', 'Average error'])
result_df
# Gamma

gamma = ['auto', 'scale']

# C

C = [0.1, 0.5, 1, 50, 100, 1000]

# epsilon

epsilon = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]



# Creating the random grid

random_grid = {'gamma': gamma,

               'C': C,

               'epsilon': epsilon

              }



# Creating an instance of `SVR`

estimator = SVR(kernel='linear', gamma='auto')

# Performing random search of parameters

rf_random = RandomizedSearchCV(estimator, random_grid,

                               n_iter=100, cv=5, verbose=2,

                               random_state=7, n_jobs=-1)

# Commenting out the below lines as the tuning takes long time to complete

# rf_random.fit(X_train, y_train)
# Get the CV results

# rf_random.cv_results_
# Get the best hyperparameters

# rf_random.best_params_
estimator = SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',

                kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

error = np.sqrt(abs(cross_val_score(estimator, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))).mean()

print(f'Model: {estimator.__class__.__name__}, Average error: {error}')



estimator.fit(X_train, y_train)

predictions = estimator.predict(X_test)

# Reversing the log transformation

predicted_saleprice = np.exp(predictions)
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predicted_saleprice.flatten()})

submission_df.to_csv('submission.csv', index=False)

FileLink('submission.csv')
submission_df.head()