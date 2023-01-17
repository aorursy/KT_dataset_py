import pandas

import matplotlib.pyplot as plt

import numpy as np

import scipy.stats

import statsmodels.api as sm

from pandas.core import datetools
df = pandas.read_csv('titanic_train.csv')
# filling empty cells with mean
def imputation(data):

    df_age= data['Age']

    mean_df = np.mean(df_age)

    data['Age'] = data['Age'].fillna(mean_df)

    return data
def features_normalize(features):

    ''' Feature Normalization When features differ by orders of magnitude,

    first performing feature scaling can make gradient descent converge 

    much more quickly'''

    sd = np.std(features, axis=0)

    mean = np.mean(features, axis=0)

    x_norm = (features - mean)/sd

    return x_norm
def compute_cost(features, values, theta):

    """

    Compute the cost of a list of parameters, theta, given a list of features 

    (input data points) and values (output data points).

    """

    m = len(values)

    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()

    cost = sum_of_square_errors / (2*m)



    return cost
def gradient_descent(features, values, theta, alpha, num_iterations):

    """

    Perform gradient descent given a data set with an arbitrary number of features.

    """



    m = len(values)

    cost_history = []

    for i in range(num_iterations):

        prediction_values = np.dot(features, theta)

        theta = theta - alpha/m * np.dot(np.transpose(features), (prediction_values-values))

        cost = compute_cost(features, values, theta)

        cost_history.append(cost)

    return theta, pandas.Series(cost_history)
def compute_r_squared(data, predictions):

    # Calculate coefficient of determination, R^2, for the model 

    mean = np.mean(data)

    sum_of_square_errors = np.square(data - predictions).sum()

    sum_of_square_data =  np.square(data - mean).sum()

    r_squared = 1- sum_of_square_errors/sum_of_square_data

    return r_squared
def prediction_grad_dic(dfa):

    df = imputation(dfa)

    

    # Select Features (try different features!)

    features = df[['Pclass', 'Age', 'SibSp', 'Parch']]

    

    

    # Add date to features using dummy variables

    dummy_units = pandas.get_dummies(df['Sex'], prefix='sex')

    features = features.join(dummy_units)

    #print dummy_units

    # Values

    values = df['Survived']

    m = len(values)

     

    features = features_normalize(features)

    add_col = features.insert(0, 'x0', 1)

#    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)

    

    # Convert features and values to numpy arrays

    features_array = np.array(features)

    values_array = np.array(values)



    # Set values for alpha, number of iterations.

    alpha = 0.05 # please feel free to change this value

    num_iterations = 100 # please feel free to change this value

    

    # Initialize theta, perform gradient descent

    theta_gradient_descent = np.zeros(len(features.columns))

    theta_gradient_descent, cost_history = gradient_descent(features_array, 

                                                            values_array, 

                                                            theta_gradient_descent, 

                                                            alpha, 

                                                            num_iterations)

 

    

    predictions = np.dot(features_array, theta_gradient_descent)

    r_squared = compute_r_squared(values_array, predictions)

    data_pred = pandas.DataFrame(data = predictions)

    feature_array = pandas.DataFrame(data = features_array)

    y = data_pred.round(decimals=0)

    x = y.astype(int)

    return x, theta_gradient_descent, cost_history
def plot_cost_history(df):

    alpha = 0.05 # please feel free to change this value

    x, theta_gradient_descent, cost_history = prediction_grad_dic(df)

    cost_df = pandas.DataFrame({

        'Cost_History': cost_history,

        'Iteration': range(len(cost_history))})

    plt.figure()

    plt.plot(cost_df['Cost_History'], cost_df['Iteration'])

   

    plt.title('Cost_History vs. Iteration for alpha = %.3f' % alpha)

    plt.xlabel('Iteration')

    plt.ylabel('Cost_History')   

    plt.show()
plot_cost_history(df)
prediction_grad_dic(df)
def data_predict(dfa):

    global df

    data = imputation(dfa)

    features = data[['Pclass', 'Age', 'SibSp', 'Parch']]

    dummy_units = pandas.get_dummies(data['Sex'], prefix='sex')

    features = features.join(dummy_units)

    features = features_normalize(features)

    add_col = features.insert(0, 'x0', 1)

    features_array = np.array(features)

    x, theta_gradient_descent, cost_history = prediction_grad_dic(df)

    predictions = np.dot(features_array, theta_gradient_descent)

    data_pred = pandas.DataFrame(data = predictions)

    y = data_pred.round(decimals=0)

    x = y.astype(int)

    p = pandas.DataFrame(data['PassengerId'])

    p.insert(1,'Survived',x)

    return p                          
data = pandas.read_csv('titanic_test.csv')
#data.dropna(subset=['Age'], inplace=True)
data_rev = data_predict(data)
data_rev.to_csv('predict_test_titanic.csv', index = False)
data_rev