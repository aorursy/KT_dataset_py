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
    
##########################
# Input train data
file_path = '../input/train.csv' 
train_data = pd.read_csv(file_path)
# Input test data
test_data = pd.read_csv('../input/test.csv')

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

# Formats train_x and test_x
train_x = np.array(train_data['GrLivArea'])
test_x = np.array(test_data['GrLivArea'])

# Process the gradient descent
print("Starting regression")
theta_gradient_descent, cost_history = gradient_descent(train_x, train_test_data, 0.03, 1000)
predictions = predict(theta_gradient_descent, test_x)
#Transform the predictions
predictions = np.exp(predictions)
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
submission.to_csv('submission_multi_gradient_uni_final.csv', index=False)
print("Done")
