# Common Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Library Settings

%matplotlib inline

pd.plotting.register_matplotlib_converters()
# Load Data to Frame

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df
# Let's look at the metadata

df.info()
drop_features = ['Id', 'GarageYrBlt', 'YrSold', 'MoSold', 'LotShape', 'LotConfig', 'LandContour', 'BldgType']

df = df.drop(drop_features, axis=1)
# Adding feature for house age

df['Age'] = 2020 - df['YearBuilt']

df['RemodelAge'] = 2020 - df['YearRemodAdd']
# Features to drop

drop_engineered_features = ['YearBuilt', 'YearRemodAdd']



# Drop features

df = df.drop(drop_engineered_features, axis=1)



# Append list to drop_features, so that test set could be set up accordingly

drop_features.extend(drop_engineered_features)
# Correlation Inspection

plt.figure(figsize=(16,10))

sns.heatmap(df.corr(), annot= False)

plt.title('Correlation')
# Visualize missing values noise

plt.figure(figsize=(22,8))

sns.heatmap(df.isnull(), yticklabels = False, cbar = False)

plt.show()
# Visualize missing values

plt.figure(figsize=(20,8))

plt.title('Percentage of Missing Data')

plt.xlabel('Features')

plt.xticks(rotation=90) 

plt.ylabel('Percentage')

plt.ylim(0, 100)

sns.barplot(x = df.columns, y = df.isnull().sum()/len(df)*100)

plt.show()
# Missing Data Summary

missing_data = pd.concat([df.isnull().sum(), df.isnull().sum()/len(df)*100], axis=1, keys=['Number of Missing', 'Percentage of Missing'])

missing_data.sort_values(by=['Percentage of Missing'], ascending=False).head(18)
# Features to drop

drop_missing_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage']



# Drop features

df = df.drop(drop_missing_features, axis=1)



# Append list to drop_features, so that test set could be set up accordingly

drop_features.extend(drop_missing_features)
# Imports

from sklearn.impute import SimpleImputer



# Setup Imputers

category_imputer = SimpleImputer(strategy='constant', fill_value='None')

numeric_imputer = SimpleImputer(strategy='constant', fill_value=0)



# Target features for Mean & Mode Imputation

category_imputer_features = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']

numeric_imputer_features = ['MasVnrArea']



# Perform Imiputations

category_imputed_features = pd.DataFrame(category_imputer.fit_transform(df[category_imputer_features]), columns = category_imputer_features)

numeric_imputed_features = pd.DataFrame(numeric_imputer.fit_transform(df[numeric_imputer_features]), columns = numeric_imputer_features)



# Drop existing features where imputations are performed

df = df.drop(category_imputer_features, axis = 1)

df = df.drop(numeric_imputer_features, axis = 1)



# Reset frame index and drop newly added index column

df = df.reset_index()

df = df.drop('index', axis = 1)



# Add imputed features to the original dataframe

df[category_imputer_features] = category_imputed_features[category_imputer_features]

df[numeric_imputer_features] = numeric_imputed_features[numeric_imputer_features]
# Drop rows with missing prices

df = df.dropna(subset=['Electrical'])
# Get list of Categorical Variables

object_columns = [column for column in df.columns if df[column].dtype == 'object']



# Get number of unique entries in each column with categorical data

object_unique = list(map(lambda column: df[column].nunique(), object_columns))

category_dictionary = dict(zip(object_columns, object_unique))



# Print number of unique entries by column, in ascending order along with cardinality

sorted(category_dictionary.items(), key=lambda x: x[1])
# Features to Inspect

inspect_features = ['MSZoning', 'Street', 'Utilities', 'CentralAir', 'Foundation', 'LandSlope', 'HouseStyle']



# Define Subplot Grid

f, axes = plt.subplots(int(np.ceil(len(inspect_features)/4)), 4, figsize=(22, 4*int(np.ceil(len(inspect_features)/4))))



# Render Plots

column_counter = 0

for index, feature in enumerate(inspect_features):

    sns.scatterplot(x=df[feature], y=df['SalePrice'], ax=axes[int(np.floor(index/4)), column_counter])

    if column_counter == 3:

        column_counter = 0

    else:

        column_counter += 1
# Features to drop

# drop_high_cardinality_features = ['Neighborhood', 'Exterior2nd', 'Exterior1st', 'SaleType', 'Condition1', 'Condition2', 'RoofMatl', 'HouseStyle', 'BsmtFinType1', 'BsmtFinType2', 'GarageType']

drop_high_cardinality_features = ['Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Heating', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']



# Drop features

df = df.drop(drop_high_cardinality_features, axis=1)



# Append list to drop_features, so that test set could be set up accordingly

drop_features.extend(drop_high_cardinality_features)
# Imports

from sklearn.preprocessing import LabelEncoder



# Features to Encode

label_features = ['Street', 'Utilities', 'CentralAir', 'Foundation', 'LandSlope']



# Apply Label Encoder 

label_encoder = LabelEncoder()

for feature in label_features:

    df[feature] = label_encoder.fit_transform(df[feature])
# Imports

from sklearn.preprocessing import OneHotEncoder



# Features to Encode ('HouseStyle')

one_hot_features = ['MSZoning']



# Initialize one-hot encoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



# Apply one-hot encoder to each column with categorical data

OH_features = pd.DataFrame(OH_encoder.fit_transform(df[one_hot_features]))



# Remove categorical columns (will replace with one-hot encoding)

numeric_features = df.drop(one_hot_features, axis=1)



# Add one-hot encoded columns to numerical features

df = pd.concat([numeric_features, OH_features], axis=1)
# Drop NaN observations

df = df.dropna()
# Final Shape of the training set

df.shape
# Imports

from sklearn.model_selection import train_test_split



# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['SalePrice'], axis = 1), df['SalePrice'], test_size = .20, random_state= 0)
# Imports

from sklearn.preprocessing import RobustScaler



# Transform

transformer = RobustScaler().fit(X_train)

X_train_scaled = transformer.transform(X_train)

X_test_scaled = transformer.transform(X_test)
# Model Imports

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



# Metrics Imports

from sklearn import metrics
# Models Object

models = {

    'LinearRegression': {

        'model': LinearRegression()

    },

    'RandomForestRegressor': {

        'model': RandomForestRegressor(n_estimators=80, max_depth=30, random_state = 0)

    },

    'XGBRegressor': {

        'model': XGBRegressor(n_estimators = 150, max_depth = 30, learning_rate = 0.1, random_state = 2)

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
# Load Data to Frame

df_test_original = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_test = df_test_original.copy()



# Process feature engineering

df_test['Age'] = 2020 - df_test['YearBuilt']

df_test['RemodelAge'] = 2020 - df_test['YearRemodAdd']



# Transform with Label Encoder

# df_test = df_test.
# Perform Imiputations

category_imputed_features_test = pd.DataFrame(category_imputer.transform(df_test[category_imputer_features]), columns = category_imputer_features)

numeric_imputed_features_test = pd.DataFrame(numeric_imputer.transform(df_test[numeric_imputer_features]), columns = numeric_imputer_features)



# Drop existing features where imputations are performed

df_test = df_test.drop(category_imputer_features, axis = 1)

df_test = df_test.drop(numeric_imputer_features, axis = 1)



# Reset frame index and drop newly added index column

df_test = df_test.reset_index()

df_test = df_test.drop('index', axis = 1)



# Add imputed features to the original dataframe

df_test[category_imputer_features] = category_imputed_features_test[category_imputer_features]

df_test[numeric_imputer_features] = numeric_imputed_features_test[numeric_imputer_features]
# Drop features not used by model

df_test = df_test.drop(drop_features, axis=1)



# Fill 0 in case of NaN's in remaining features

df_test = df_test.fillna(0)
# Unknown/New Categories

label_encoder.classes_ = np.append(label_encoder.classes_, '<unknown>')



# Transform with Label Encoder

for feature in label_features:

    

    # Handle new categories in test set

    df_test[feature] = df_test[feature].map(lambda s: '<unknown>' if s not in label_encoder.classes_ else s)

    

    # Transform

    df_test[feature] = label_encoder.transform(df_test[feature])
# Transform with One-Hot Encoder

OH_test = pd.DataFrame(OH_encoder.transform(df_test[one_hot_features]))

OH_test.index = df_test.index

df_test = df_test.drop(one_hot_features, axis=1)

df_test = pd.concat([df_test, OH_test], axis=1)
# Apply Scaling

df_test_scaled = transformer.transform(df_test)
# Generate Predictions

predictions = models['XGBRegressor']['model'].predict(df_test_scaled)
# Generate Output Frame

df_predictions = pd.DataFrame({'Id': df_test_original.Id, 'SalePrice': predictions})
# Submit results

df_predictions.to_csv('submission.csv', index=False)