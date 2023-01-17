import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline

import warnings

warnings.filterwarnings('ignore')



from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
# Path of the file to read.

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)



# path to file you will use for predictions

test_data = pd.read_csv('../input/test.csv')
print(home_data.shape)

print(test_data.shape)
home_data.head(3)
test_data.head(3)
home_data.columns
home_data.get_dtype_counts()
test_data.get_dtype_counts()
home_num = home_data.select_dtypes(exclude = 'object')

print(home_num.shape)

home_num.describe()
home_object = home_data.select_dtypes(exclude = ['int64', 'float64'])

print(home_object.shape)

home_object.describe()
test_num = test_data.select_dtypes(exclude = 'object')

print(test_num.shape)

test_num.describe()
test_object = test_data.select_dtypes(exclude = ['int64', 'float64'])

print(test_object.shape)

test_object.describe()
home_num.isnull().sum()
test_num.isnull().sum()
home_num.drop('GarageYrBlt', axis = 1, inplace = True)

test_num.drop('GarageYrBlt', axis = 1, inplace = True)
home_data.LotFrontage.describe()
# filling the null values in LotFrontage with the median values

home_num.LotFrontage.fillna(home_data.LotFrontage.median(), inplace = True)

test_num.LotFrontage.fillna(test_data.LotFrontage.median(), inplace = True)
plt.figure(figsize = (8,5))

sns.distplot(home_num.SalePrice, kde = False)

print("Skew is : ", home_num.SalePrice.skew())
# to reduce skewness

print("Skew is : ", np.log(home_num.SalePrice).skew())

sns.distplot(np.log(home_num.SalePrice))
plt.figure(figsize = (12,18))

num = home_num.drop('SalePrice', axis = 1)

for i in range(len(num.columns)):

    plt.subplot(9,4,i+1)

    sns.boxplot(y = num.iloc[:,i].dropna())

    plt.xlabel(num.columns[i])

    

plt.tight_layout()

plt.show()
correlation = home_num.corr()

plt.figure(figsize = (12,10))

sns.heatmap(correlation)
correlation['SalePrice'].sort_values(ascending=False)
home_num.drop(['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold', 'MoSold',

              'MSSubClass', 'EnclosedPorch', 'KitchenAbvGr', '3SsnPorch'], axis = 1, inplace = True)

print(home_num.shape)



test_num.drop(['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold', 'MoSold',

              'MSSubClass', 'EnclosedPorch', 'KitchenAbvGr', '3SsnPorch'], axis = 1, inplace = True)

print(test_num.shape)
home_object.isnull().sum().sort_values(ascending = False)
home_object.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish',

                 'Utilities', 'Condition2', 'RoofMatl', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 

                  'BsmtCond','MasVnrType', 'Electrical', 'Exterior1st', 'Exterior2nd', 'HouseStyle', 'Heating'],

                 axis = 1 , inplace = True)
test_object.isnull().sum().sort_values(ascending = False)
test_object.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish',

                 'Utilities', 'Condition2', 'RoofMatl', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 

                  'BsmtCond','MasVnrType', 'Electrical', 'Exterior1st', 'Exterior2nd', 'HouseStyle', 'Heating'],

                 axis = 1 , inplace = True)
home_data = pd.concat([home_num, home_object], axis = 1)

home_data.shape
home_data.columns
test_data = pd.concat([test_num, test_object], axis = 1)

test_data.shape
test_data.columns


# Create target object and call it y

y = home_data.SalePrice



# Create X

X = home_data.drop('SalePrice', axis = 1)

X = pd.get_dummies(X)



# Final imputation of missing data - to address those outstanding after previous section

myimputer = SimpleImputer()

X = myimputer.fit_transform(X)

X.shape
# Split into validation and training data

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
reg = LinearRegression().fit(train_X,train_y)

pred = reg.predict(test_X)

val_mae = mean_absolute_error(pred, test_y)

print("Validation MAE: {:,.0f}".format(val_mae))
# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)



# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(test_X)

val_mae = mean_absolute_error(val_predictions, test_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(test_X)

val_mae = mean_absolute_error(val_predictions, test_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(test_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, test_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state = 1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

test = test_data.copy()

test_data = pd.get_dummies(test_data)

test_data = myimputer.transform(test_data)

test_preds = rf_model_on_full_data.predict(test_data)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)