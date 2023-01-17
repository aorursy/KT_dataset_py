import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
pd.options.mode.chained_assignment = None  # default='warn'


#######################  LINEAR REGRESSION ###########################
def predict(theta, x_test):
    if len(x_test.shape) < 2:
        x_test = x_test.reshape(-1,1)
        x_test = np.insert(x_test, 0, 1, 1)
    return np.dot(x_test, theta) 

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
########################## MISSING VALUES #########################################
def handle_missing_values(data):
    for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
            'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
        train_data[col] = train_data[col].fillna('None')
        test_data[col] = test_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna(0)
    train_data["LotFrontage"] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
    test_data["LotFrontage"] = test_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
    
##########################
# Input train data
file_path = '../input/train.csv' 
train_data = pd.read_csv(file_path)
# Input test data
test_data = pd.read_csv('../input/test.csv')

# Handle the missing values
handle_missing_values(train_data)
handle_missing_values(test_data)

# Removing GrLivArea > 4000 square feet area and eletrical input from train data
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
train_data = train_data[train_data.GrLivArea < 4000]

# Log Transformation of GrLivArea and SalePrice
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
test_data['GrLivArea'] = np.log(test_data['GrLivArea'])
train_data['SalePrice'] = np.log(train_data['SalePrice'])

# Spliting features and values from train data
train_test_data = train_data['SalePrice']
train_data = train_data.drop(['SalePrice'], axis = 1)

# One-hot enconding for both train and test data. The data sets are concatenated, encoded and splited again
train_objs_num = len(train_data)
dataset = pd.concat(objs=[train_data, test_data], axis=0, sort=False)
dataset_preprocessed = pd.get_dummies(dataset)
train_data = dataset_preprocessed[:train_objs_num]
test_data = dataset_preprocessed[train_objs_num:]

# Log Transformation of all numerical variables with skewness > 0.5
categorical_features = train_data.select_dtypes(include = ["object"]).columns
train_data_categorical = train_data[categorical_features]
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
train_data_numerical = train_data[numerical_features]
skewness = train_data_numerical.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
train_data_numerical[skewed_features] = np.log1p(train_data_numerical[skewed_features])
# Test data transformation 
test_data[skewed_features] = np.log1p(test_data[skewed_features])

# Join categorical and numerical variables
train_data = pd.concat([train_data_numerical, train_data_categorical], axis = 1)

# Formats train_x and test_x
train_x = np.array(train_data)
test_x = np.array(test_data)

# Process the gradient descent
print("Starting regression")
theta_gradient_descent, cost_history = gradient_descent(train_x, train_test_data, 0.00000003, 1000)
predictions = predict(theta_gradient_descent, test_data)
#Transform the preditions back
predictions = np.exp(predictions)
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
submission.to_csv('submission_multi_gradient_final2.csv', index=False)
print("Done")

