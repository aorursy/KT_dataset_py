import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

####################### COMPARISON METHOD - RANDOM FOREST ############################
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
    theta = abs(np.random.normal(0,0.02, size = features[0].shape))
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

# 1st Test Scenario - UNIVARIATE REGRESSION
# GrLivArea, n_iterations = 1000, alpha = 0.0000001
print("1st Gradient Test - GrLivArea - RMSE = ", process_regression_and_test(train_data['GrLivArea'], 
                                             train_data['SalePrice'], 
                                             0.0000001, 
                                             1000, 
                                             0.25))
print("1st RandomForest Test - GrLivArea - RMSE = ", process_random_forest_comparrison(train_data['GrLivArea'],
                                                                                      train_data['SalePrice'],
                                                                                      0.25))
# OverallQual, n_iterations = 1000, alpha = 0.01
print("1st Gradient Test - OverallQual - RMSE = ", process_regression_and_test(train_data['OverallQual'], 
                                             train_data['SalePrice'], 
                                             0.01, 
                                             1000, 
                                             0.25))
print("1st RandomForest Test - OverallQual - RMSE = ", process_random_forest_comparrison(train_data['OverallQual'],
                                                                                      train_data['SalePrice'],
                                                                                      0.25))
# 2nd Test Scenario - UNIVARIATE REGRESSION - After outliers removal and log transformation of GrLivArea and SalePrice
# Removing GrLivArea > 4000 square feet area
train_data = train_data[train_data.GrLivArea < 4000]

# Log Transformation of GrLivArea and SalePrice
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
train_data['SalePrice'] = np.log(train_data['SalePrice'])
# GrLivArea, n_iterations = 1000, alpha = 0.03
print("2nd Test - GrLivArea - RMSE = ", process_regression_and_test(train_data['GrLivArea'], 
                                             train_data['SalePrice'], 
                                             0.03, 
                                             1000, 
                                             0.25))
print("2nd RandomForest Test - GrLivArea - RMSE = ", process_random_forest_comparrison(train_data['GrLivArea'],
                                                                                      train_data['SalePrice'],
                                                                                      0.25))
# OverallQual, n_iterations = 1000, alpha = 0.03
print("2nd Gradient Test - OverallQual - RMSE = ", process_regression_and_test(train_data['OverallQual'], 
                                             train_data['SalePrice'], 
                                             0.03, 
                                             1000, 
                                             0.25))
print("2nd RandomForest Test - OverallQual - RMSE = ", process_random_forest_comparrison(train_data['OverallQual'],
                                                                                      train_data['SalePrice'],
                                                                                      0.25))
