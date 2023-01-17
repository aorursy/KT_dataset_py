# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
input_directory = '/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv'
input_dataset = pd.read_csv(input_directory)
input_dataset.shape
input_dataset.head()
input_dataset.dtypes
input_dataset['floor'].value_counts()
input_dataset['floor'] = input_dataset['floor'].replace('-',0)
input_dataset['floor'].value_counts()
input_dataset['floor'] = pd.to_numeric(input_dataset['floor'])
input_dataset.isnull().sum()
input_dataset['animal'].value_counts()
input_dataset['furniture'].value_counts()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
input_dataset['city'] = encoder.fit_transform(input_dataset['city'])

input_dataset['animal'] = encoder.fit_transform(input_dataset['animal'])

input_dataset['furniture'] = encoder.fit_transform(input_dataset['furniture'])
input_dataset.dtypes
sns.heatmap(input_dataset.corr())
plt.plot(input_dataset['total (R$)'], input_dataset['hoa (R$)'], 'o')

plt.suptitle('total VS hoa')
# Let us remove the outliers

input_dataset.sort_values('hoa (R$)', ascending = False).head(10)

# the Indexes - [255, 6979, 6230, 2859, 2928, 1444] are outliers. Let us remove them
input_dataset = input_dataset.drop([255, 6979, 6230, 2859, 2928, 1444])
input_dataset.sort_values('hoa (R$)', ascending = False).head()
plt.plot(input_dataset['total (R$)'], input_dataset['hoa (R$)'], 'o')

plt.suptitle('total VS hoa (after removing outliers)')
input_dataset.sort_values('total (R$)', ascending = False).head()
input_dataset = input_dataset.drop([6645, 2182])
plt.plot(input_dataset['total (R$)'], input_dataset['hoa (R$)'], 'o')

plt.suptitle('total VS hoa (after removing outliers)')
plt.plot(input_dataset['rent amount (R$)'], input_dataset['fire insurance (R$)'], 'o')

plt.suptitle('rent amount VS fire insurance')
input_dataset.sort_values('rent amount (R$)',ascending = False).head()
input_dataset.sort_values('fire insurance (R$)',ascending = False).head()
input_dataset = input_dataset.drop([7748])
plt.plot(input_dataset['rent amount (R$)'], input_dataset['fire insurance (R$)'], 'o')

plt.suptitle('rent amount VS fire insurance')
# separating training and testing data

from sklearn.model_selection import train_test_split

inputs = input_dataset.drop(columns = 'rent amount (R$)')

target = input_dataset['rent amount (R$)']

train_input, test_input, train_target, test_target = train_test_split(inputs, target, test_size = 0.2, random_state = 101)
print(train_input.shape)

print(train_target.shape)

print(test_input.shape)

print(test_target.shape)
# importing machine learning algorithms

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor
# importing evaluation metrics for regression algorithms

from sklearn.metrics import mean_squared_error
# KNN Algorithm

knn = KNeighborsRegressor(n_neighbors = 2)

model = knn.fit(train_input, train_target)

print('model = ', model)

prediction = knn.predict(test_input)

print(np.sqrt(mean_squared_error(test_target, prediction)))
# Logistic Algorithm

logit = LogisticRegression()

model = logit.fit(train_input, train_target)

print('model = ',model)

prediction = model.predict(test_input)

print(np.sqrt(mean_squared_error(test_target, prediction)))
# Linear Regression Algorithm

linear = LinearRegression()

model = linear.fit(train_input, train_target)

print('model = ',model)

prediction = model.predict(test_input)

print(np.sqrt(mean_squared_error(test_target, prediction)))
# Support Vector Machine

svm = SVR()

model = svm.fit(train_input, train_target)

print('model = ', model)

prediction = model.predict(test_input)

print(np.sqrt(mean_squared_error(test_target, prediction)))
# Multilayered Perceptron

mlp = MLPRegressor()

model = mlp.fit(train_input, train_target)

print('model = ', model)

prediction = model.predict(test_input)

print(np.sqrt(mean_squared_error(test_target, prediction)))