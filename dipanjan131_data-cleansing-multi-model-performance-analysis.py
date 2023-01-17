# Imports

import seaborn as sns

import pandas as pd

import numpy as np

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline
# Load Data to Frame; Column #7 contains dates

df = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv', parse_dates = [7])
df
df.info()
# Adding feature for house age

df['Age'] = 2020 - df['YearBuilt']
# Drop unnecessary features

df = df.drop(['Address', 'Postcode', 'Method', 'SellerG', 'Propertycount', 'Date', 'YearBuilt', 'Bedroom2'], axis = 1)
df
df.info()
# Missing Values per Feature

df.isnull().sum()
# Visualize missing values noise

plt.figure(figsize=(15,8))

sns.heatmap(df.isnull(), yticklabels = False, cbar = False)

plt.show()
# Visualize missing values

plt.figure(figsize=(20,8))

plt.title('Percentage of Missing Data')

plt.xlabel('Features')

plt.ylabel('Percentage')

plt.ylim(0, 100)

sns.barplot(x = df.columns, y = df.isnull().sum()/len(df)*100)

plt.show()
# Correlation Inspection

plt.figure(figsize=(16,10))

sns.heatmap(df.corr(), annot= True)

plt.title('Correlation')
# Drop rows with missing prices

df = df.dropna(subset=['Price'])



# Drop feature BuildingArea - Ignore this, models perform better with this feature imputed with mean

# df = df.drop(['BuildingArea'], axis = 1)
# Imports

from sklearn.model_selection import train_test_split



# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Price', axis = 1), df['Price'], test_size = .20, random_state= 0)
# Imports

from sklearn.impute import SimpleImputer



# Setup Imputers

mean_imputer = SimpleImputer(strategy='mean')

mode_imputer = SimpleImputer(strategy='most_frequent')



# Target features for Mean & Mode Imputation

mean_imputer_features = ['Landsize', 'Age', 'Lattitude', 'Longtitude', 'BuildingArea']

mode_imputer_features = ['Car', 'Bathroom']



# Perform Imiputations

mean_imputed_features_train = pd.DataFrame(mean_imputer.fit_transform(X_train[mean_imputer_features]), columns = mean_imputer_features)

mean_imputed_features_test = pd.DataFrame(mean_imputer.transform(X_test[mean_imputer_features]), columns = mean_imputer_features)

mode_imputed_features_train = pd.DataFrame(mean_imputer.fit_transform(X_train[mode_imputer_features]), columns = mode_imputer_features)

mode_imputed_features_test = pd.DataFrame(mean_imputer.transform(X_test[mode_imputer_features]), columns = mode_imputer_features)
# Drop existing features where imputations are performed

X_train = X_train.drop(mean_imputer_features, axis = 1)

X_train = X_train.drop(mode_imputer_features, axis = 1)

X_test = X_test.drop(mean_imputer_features, axis = 1)

X_test = X_test.drop(mode_imputer_features, axis = 1)



# Reset frame index and drop newly added index column

X_train = X_train.reset_index()

X_train = X_train.drop('index', axis = 1)

X_test = X_test.reset_index()

X_test = X_test.drop('index', axis = 1)



# Add imputed features to the original dataframe

X_train[mean_imputer_features] = mean_imputed_features_train[mean_imputer_features]

X_train[mode_imputer_features] = mode_imputed_features_train[mode_imputer_features]

X_test[mean_imputer_features] = mean_imputed_features_test[mean_imputer_features]

X_test[mode_imputer_features] = mode_imputed_features_test[mode_imputer_features]
# Merge Training Features and Labels

y_train = y_train.reset_index()

y_train.drop('index', axis=1)

df_train = X_train.copy()

df_train['Price'] = y_train['Price']

df_train
df_train.isnull().sum()
# Drop rows with missing values

df_train = df_train.dropna()
# Split back the training Features and Labels

X_train = df_train.drop('Price', axis=1)

y_train = df_train['Price']
# Get list of Categorical Variables

object_columns = [column for column in X_train.columns if X_train[column].dtype == 'object']
# Get number of unique entries in each column with categorical data

object_unique = list(map(lambda column: X_train[column].nunique(), object_columns))

dictionary = dict(zip(object_columns, object_unique))
# Print number of unique entries by column, in ascending order along with cardinality

sorted(dictionary.items(), key=lambda x: x[1])
# Drop high cardinality features

X_train = X_train.drop(['CouncilArea', 'Suburb'], axis=1)

X_test = X_test.drop(['CouncilArea', 'Suburb'], axis=1)
# Imports

from sklearn.preprocessing import OneHotEncoder



# Features to encode

one_hot_features = ['Type', 'Regionname']



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_train = pd.DataFrame(OH_encoder.fit_transform(X_train[one_hot_features]))

OH_test = pd.DataFrame(OH_encoder.transform(X_test[one_hot_features]))
# One-hot encoding removed index; put it back

OH_train.index = X_train.index

OH_test.index = X_test.index



# Remove categorical columns (will replace with one-hot encoding)

X_train = X_train.drop(one_hot_features, axis=1)

X_test = X_test.drop(one_hot_features, axis=1)



# Add one-hot encoded columns to numerical features

X_train = pd.concat([X_train, OH_train], axis=1)

X_test = pd.concat([X_test, OH_test], axis=1)
X_train
# Imports

from sklearn.preprocessing import RobustScaler
# Transform

transformer = RobustScaler().fit(X_train)

X_train_scaled = transformer.transform(X_train)

X_test_scaled = transformer.transform(X_test)
# Imports

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor



from sklearn import metrics
# Models Object

models = {

    'Lasso': {

        'model': Lasso()

    },

    'LinearRegression': {

        'model': LinearRegression()

    },

    'Ridge': {

        'model': Ridge()

    },

    'ElasticNet': {

        'model': ElasticNet()

    },

    'KNeighborsRegressor': {

        'model': KNeighborsRegressor()

    },

    'RandomForestRegressor': {

        'model': RandomForestRegressor()

    },

    'GradientBoostingRegressor': {

        'model': GradientBoostingRegressor()

    },

    'AdaBoostRegressor': {

        'model': AdaBoostRegressor(n_estimators = 5, learning_rate = 1.2, loss = 'exponential', random_state = 2)

    },

    'DecisionTreeRegressor': {

        'model': DecisionTreeRegressor(max_depth = 9, min_samples_split = 4, random_state = 1)

    },

    'XGBRegressor': {

        'model': XGBRegressor(n_estimators = 80, max_depth = 8, learning_rate = 0.3, random_state = 2)

    }

}
# Add dictionary attributes

for model in models:

    models[model]['prediction'] = None

    models[model]['errors'] = {

        'mae': None,

        'mse': None,

        'rmse': None

    }

    models[model]['scores'] = {

        'r2': None

    }
# Let's try our luck with a bunch of models

for model in models:

    print('Running ', models[model]['model'])

    models[model]['model'].fit(X_train_scaled, y_train)

    models[model]['predictions'] = models[model]['model'].predict(X_test_scaled)

    models[model]['errors']['mae'] = metrics.mean_absolute_error(y_test, models[model]['predictions'])

    models[model]['errors']['mse'] = metrics.mean_squared_error(y_test, models[model]['predictions'])

    models[model]['errors']['rmse'] = np.sqrt(models[model]['errors']['mse'])

    models[model]['scores']['r2'] = metrics.r2_score(y_test, models[model]['predictions'])

    print('MAE: ', models[model]['errors']['mae'])

    print('MSE: ', models[model]['errors']['mse'])

    print('RMSE: ', models[model]['errors']['rmse'])

    print('R2: ', models[model]['scores']['r2'])

    print('\n')
# Analyse the Residuals

for index, model in enumerate(models):

    sns.scatterplot(models[model]['predictions'], y_test)

    plt.title(model)

    plt.show()

    sns.distplot((y_test - models[model]['predictions']))

    plt.title(model)

    plt.show()