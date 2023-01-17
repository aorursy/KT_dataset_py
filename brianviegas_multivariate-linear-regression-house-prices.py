import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
pd.options.mode.chained_assignment = None  # default='warn'

####################### COMPARISON METHODS ############################
def process_random_forest_comparrison(train_data, target_data, test_size_value):
    # Splits train and test data
    train_x, test_x, train_y, test_y = train_test_split(train_data, 
                                                        target_data, 
                                                        test_size = test_size_value, 
                                                        random_state = None)
    # Formats data
    array_train_x = np.array(train_x)
    array_train_y = np.array(train_y)
    array_test_x = np.array(test_x)
    if len(train_x.shape) < 2:
        array_train_x = array_train_x.reshape(-1,1)
        array_test_x = array_test_x.reshape(-1,1)
    # Train the model    
    model = RandomForestRegressor()
    model.fit(array_train_x, array_train_y)
    # Process the predictions
    predictions = model.predict(array_test_x)
    # Returns the RMSE evaluation
    return root_mean_squared_error(test_y, predictions)

#######################  LINEAR REGRESSION ###########################
def process_regression_and_test(train_data, target_data, alpha, number_iterations, test_size_value):
    # Splits train and test data
    train_x, test_x, train_y, test_y = train_test_split(train_data, 
                                                        target_data, 
                                                        test_size = test_size_value, 
                                                        random_state = None)
    # Formats train_x and test_x
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    # Process the gradient descent
    theta_gradient_descent, cost_history = gradient_descent(train_x, train_y, alpha, number_iterations)
    # Process the predictions
    predictions = predict(theta_gradient_descent, test_x)
    # Returns the RMSE evaluation
    return root_mean_squared_error(test_y, predictions)

def predict(theta, x_test):
    if len(x_test.shape) < 2:
        x_test = x_test.reshape(-1,1)
        x_test = np.insert(x_test, 0, 1, 1)
    return np.dot(x_test, theta) 

######################## EVALUATION FUNCTIONS #############################
def mean_squared_error(data, predictions):
    error = data - predictions
    return np.mean(error**2)

def root_mean_squared_error(data, predictions):
    return np.sqrt(mean_squared_error(data, predictions))

######################### GRADIENT DESCENT ALGORITHM ####################
def compute_cost(features, values, theta):
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)
    return cost

def gradient_descent(features, values, alpha, num_iterations):
    if len(features.shape) < 2:
        features = features.reshape(-1,1)
        features = np.insert(features, 0, 1, 1)    
    theta = abs(np.random.normal(0,0.00002, size = features[0].shape))
    m = len(values)
    cost_history = []
    for i in range(num_iterations):
        predict_values = np.dot(features, theta)
        theta = theta - (alpha / m * np.dot((predict_values - values), features))
        cost_theta = compute_cost(features, values, theta)
        cost_history.append(cost_theta)
    return theta, pd.Series(cost_history)

########################## END OF GRADIENT DESCENT ALGORITHM #######################
# Input data
file_path = '../input/train.csv' 
train_data = pd.read_csv(file_path)
second_train_data = train_data.copy()
# Handle the missing values
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
            'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
    train_data[col] = train_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    train_data[col] = train_data[col].fillna(0)
train_data["LotFrontage"] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
train_data = pd.get_dummies(train_data)

# 1st Test Scenario - MULTIVARIATE REGRESSION -  Missing values handled and one-hot enconding
# n_iterations = 1000, alpha = 0.0000001
print("1st Gradient Test - All Variables - RMSE = ", process_regression_and_test(train_data.drop(['SalePrice'], axis = 1), 
                                             train_data['SalePrice'], 
                                             0.00000000001, 
                                             1000, 
                                             0.25))
print("1st RandomForest Test - All Variables - RMSE = ", process_random_forest_comparrison(train_data.drop(['SalePrice'], axis = 1),
                                                                                      train_data['SalePrice'],
                                                                                      0.25))
# 2nd Test Scenario - MULTIVARIATE REGRESSION - After outliers removal, missing values handled, log transformations and one-hot enconding
# Handle the missing values
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
            'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
    second_train_data[col] = second_train_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    second_train_data[col] = second_train_data[col].fillna(0)
second_train_data["LotFrontage"] = second_train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
second_train_data = second_train_data.drop(second_train_data.loc[second_train_data['Electrical'].isnull()].index)

# Removing GrLivArea > 4000 square feet area
second_train_data = second_train_data[second_train_data.GrLivArea < 4000]

# Log Transformation of GrLivArea and SalePrice
second_train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
second_train_data['SalePrice'] = np.log(train_data['SalePrice'])

# Log Transformation of all numerical variables with skewness > 0.5
numerical_features = second_train_data.select_dtypes(exclude = ["object"]).columns
train_data_numerical = second_train_data[numerical_features]
skewness = train_data_numerical.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
train_data_numerical[skewed_features] = np.log1p(train_data_numerical[skewed_features])

# One-hot enconding of all categorical variables
categorical_features = second_train_data.select_dtypes(include = ["object"]).columns
train_data_categorical = second_train_data[categorical_features]
train_data_categorical = pd.get_dummies(train_data_categorical)

# Join categorical and numerical variables
second_train_data = pd.concat([train_data_numerical, train_data_categorical], axis = 1)
# n_iterations = 1000, alpha = 0.00000003
print("2nd Gradient Test - All Variables - RMSE = ", process_regression_and_test(second_train_data.drop(['SalePrice'], axis = 1), 
                                             second_train_data['SalePrice'], 
                                             0.00000003, 
                                             1000, 
                                             0.25))
print("2nd RandomForest Test - All Variables - RMSE = ", process_random_forest_comparrison(second_train_data.drop(['SalePrice'], axis = 1),
                                                                                      second_train_data['SalePrice'],
                                                                                      0.25))


