# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Read the dataset

train = pd.read_csv('../input/random-linear-regression/train.csv')

test = pd.read_csv('../input/random-linear-regression/test.csv')



print("TrainData: {}, TestData: {}".format(train.shape, test.shape))



#Clean dataset for null values(if any)

train = train.dropna()



#Store data and labels separately

train_data = np.array(train.x).reshape(train.shape[0], 1)

train_labels = np.array(train.y).reshape(train.shape[0], 1)



test_data = np.array(test.x).reshape(test.shape[0], 1)

test_labels = np.array(test.y).reshape(test.shape[0], 1)





#Set Hyperparameter values

iterations = 20

learning_rate = 0.0001



#Random initializations

weight = np.random.uniform(0, 1) * -1

bias = np.random.uniform(0, 1) * -1

plt.figure()



#Training

for iteration in range(iterations):

    

    #Forward propagation

    predictions = np.multiply(weight, train_data) + bias

    

    #CostFunction(MSE: Mean Squared Error)

    cost = np.mean((train_labels - predictions) ** 2) * 0.5

    print("Iteration: {}, Loss: {}".format(iteration+1, cost))

    

    #Plot the current status

    plt.plot(train_data, train_labels, '.')

    plt.plot(train_data, predictions, linewidth = 2)

    plt.show()

    

    #Gradient Descent for back propagation

    cost_derivative = (train_labels - predictions) * -1

    derivative_wrt_weight = np.mean(np.multiply(train_data, cost_derivative))

    derivative_wrt_bias = np.mean(cost_derivative)

    

    #Update weight & bias parameters

    weight = weight - learning_rate * derivative_wrt_weight

    bias = bias - learning_rate * derivative_wrt_bias



#Testing

test_predictions = test_data * weight + bias

cost_test = np.mean((test_labels - test_predictions) ** 2) * 0.5

print("Model performance on test data \nCost: {}".format(cost_test))



plt.figure()

plt.plot(test_data, test_labels, '.')

plt.plot(test_data, test_predictions, linewidth = 2)

plt.show()