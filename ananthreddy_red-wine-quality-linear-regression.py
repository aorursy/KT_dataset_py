# import the following packages pandas, numpy and sklearn to perform prediction.
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data-set.
df = pd.read_csv('../input/winequality-red.csv')

# Quality the parameter to be predicted is represented as X.
X = df[['quality']]
# All the input parameters used to predict the value are represented as y.
y = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

# Data-set is divided into test data and train data based on test_size variable.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Model is predicted with Linear Regression. Fit the training data-set and predict the values of the test data-set.
model = linear_model.LinearRegression()
model = model.fit(y_train, X_train)
predicted_data = model.predict(y_test)
predicted_data = np.round_(predicted_data)

# Calculate the Mean Squared Error between the predicted data and the actual data.
print (mean_squared_error(X_test,predicted_data))

# Print the predicted data.
print (predicted_data)