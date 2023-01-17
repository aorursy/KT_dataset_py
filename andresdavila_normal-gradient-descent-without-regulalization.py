# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, r2_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import data and get intuition

data = pd.read_csv("/kaggle/input/insurance/insurance.csv")



print(data.describe(), '\n', data.keys())
# Substitue binary genders

data['sex'] = [1 if ele == 'female' else 0 for ele in data['sex'].values]

# Substitute binary smoker

data['smoker'] = [1 if ele == 'yes' else 0 for ele in data['smoker'].values]

sns.heatmap(data=data.corr(), annot=True)

plt.show()
# Split data into testing, validation and training

training_data = data[len(data)//3:]

testing_data = data[:len(data) // 3]

validation_data = training_data[:len(training_data) // 2]



# Seperate label and features

x_training = training_data[['bmi', 'age', 'smoker']]

y_training = np.array(training_data['charges']).reshape(len(training_data), 1)

x_validation = validation_data[['bmi', 'age', 'smoker']]

y_validation = np.array(validation_data['charges']).reshape(len(validation_data), 1)

x_testing = testing_data[['bmi', 'age', 'smoker']]

y_testing = np.array(testing_data['charges']).reshape(len(validation_data), 1)



# Scale features

scaler = StandardScaler()

scaler.fit(x_training, y_training)

x_training = scaler.transform(x_training)

scaler.fit(x_validation, y_validation)

x_validation = scaler.transform(x_validation)

scaler.fit(x_testing, y_testing)

x_testing = scaler.transform(x_testing)
# Perform Gradient Descent 

m = x_training.shape[0]

# Add bias term to x

x_training = np.c_[np.ones(shape=(x_training.shape[0], 1)), x_training]

theta = np.random.randn(x_training.shape[1], 1)

iterations = 10000

learning_rate = 0.001

cost_history = np.zeros(shape=(iterations, 1))



for i in range(iterations):

        cost_history[i] = 1 / (2 * m) * np.sum((x_training.dot(theta) - y_training) ** 2)

        gradient = 1/m * (x_training.T.dot(np.dot(x_training, theta) - y_training))

        theta -= learning_rate*gradient
# Plot the cost function over the epochs

plt.plot(range(len(cost_history)), cost_history)

plt.show()
# Add bias node

x_validation = np.c_[np.ones(shape=(x_validation.shape[0], 1)), x_validation]

x_testing = np.c_[np.ones(shape=(x_testing.shape[0], 1)), x_testing]



y_pred_validation = x_validation.dot(theta)

y_pred_training = x_training.dot(theta)

y_pred_testing = x_testing.dot(theta)
print(f'MAE Training: {mean_absolute_error(y_pred=y_pred_training, y_true=y_training)}')

print(f'MAE Validation: {mean_absolute_error(y_pred=y_pred_validation, y_true=y_validation)}')

print(f'MAE Testing: {mean_absolute_error(y_pred=y_pred_testing, y_true=y_testing)}')
print(f'Correlation Coefficient for the testing data: {r2_score(y_testing, y_pred_testing)}')
# Split data into testing, validation and training

training_data = data[len(data)//3:]

testing_data = data[:len(data) // 3]

validation_data = training_data[:len(training_data) // 2]



# Seperate label and features

x_training = training_data[['bmi', 'age', 'smoker']].values

y_training = np.array(training_data['charges']).reshape(len(training_data), 1)

x_validation = validation_data[['bmi', 'age', 'smoker']].values

y_validation = np.array(validation_data['charges']).reshape(len(validation_data), 1)

x_testing = testing_data[['bmi', 'age', 'smoker']].values

y_testing = np.array(testing_data['charges']).reshape(len(validation_data), 1)



# Add powers of the most correlated feature

# bmi = x_training[:, 0]**2

age = x_training[:, 1] **2

x_training = np.c_[x_training, age]



# bmi = x_validation[:, 0]**2

age = x_validation[:, 1] **2

x_validation = np.c_[x_validation, age]



# bmi = x_testing[:, 0]**2

age = x_testing[:, 1] **2

x_testing = np.c_[x_testing, age]



# Scale features

scaler = StandardScaler()

scaler.fit(x_training, y_training)

x_training = scaler.transform(x_training)

scaler.fit(x_validation, y_validation)

x_validation = scaler.transform(x_validation)

scaler.fit(x_testing, y_testing)

x_testing = scaler.transform(x_testing)

print(x_testing)
# Perform Gradient Descent 

m = x_training.shape[0]

# Add bias term to x

x_training = np.c_[np.ones(shape=(x_training.shape[0], 1)), x_training]

theta = np.random.randn(x_training.shape[1], 1)

iterations = 10000

learning_rate = 0.001

cost_history = np.zeros(shape=(iterations, 1))

for i in range(iterations):

        cost_history[i] = 1 / (2 * m) * np.sum((x_training.dot(theta) - y_training) ** 2)

        gradient = 1/m * (x_training.T.dot(np.dot(x_training, theta) - y_training))

        theta -= learning_rate*gradient
# Plot the cost function over the epochs

plt.plot(range(len(cost_history)), cost_history)

plt.show()
# Add bias node

x_validation = np.c_[np.ones(shape=(x_validation.shape[0], 1)), x_validation]

x_testing = np.c_[np.ones(shape=(x_testing.shape[0], 1)), x_testing]



y_pred_validation = x_validation.dot(theta)

y_pred_training = x_training.dot(theta)

y_pred_testing = x_testing.dot(theta)



# Print Metrics

print(f'MAE Training: {mean_absolute_error(y_pred=y_pred_training, y_true=y_training)}')

print(f'MAE Validation: {mean_absolute_error(y_pred=y_pred_validation, y_true=y_validation)}')

print(f'MAE Testing: {mean_absolute_error(y_pred=y_pred_testing, y_true=y_testing)}')
print(f'Correlation Coefficient for the testing data: {r2_score(y_testing, y_pred_testing)}')