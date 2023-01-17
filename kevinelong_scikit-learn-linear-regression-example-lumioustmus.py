import matplotlib.pylab as plt
import numpy as np
import pandas as pd
%matplotlib inline
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import datasets
diabetes = datasets.load_diabetes() # load data

print( diabetes.DESCR )
#diabetes.DESCR
diabetes.data.shape # feature matrix shape
diabetes.target.shape # target vector shape
diabetes.feature_names # column names
# print(type(diabetes.data))
# diabetes.data = sorted(diabetes.data, key = lambda x: x[-1])
# print(type(diabetes.data))
# print(diabetes.data)

# Sperate train and test data
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.5, random_state=0)
# Split the data into training/testing sets
X_train = diabetes.data[:-20]
X_test = diabetes.data[-20:]

# Split the targets into training/testing sets
y_train = diabetes.data[:-20]
y_test = diabetes.data[-20:]
# Sperate train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
# There are three steps to model something with sklearn
# 1. Set up the model
model = LinearRegression()
# model = LogisticRegression()
# 2. Use fit
model.fit(X_train, y_train)
# 3. Check the score
model.score(X_test, y_test)

# print(model.__dict__)

model.coef_ # Get the coefficients, beta
model.intercept_ # Get the intercept, c
model.predict(X_test) # Predict unkown data
# plot prediction and actual data
y_pred = model.predict(X_test) 

print(model.intercept_)
print(model.coef_)

print("Xt", X_test[0])
print("yt", y_test[0])
print("yp", y_pred[0])

print(y_pred[0] / y_test[0])
# print(y_test[0] / y_pred[0])
output = []

for x,B in zip(X_test[0],model.coef_):
    output.append(x * B)
# print(output)    
# print(sum(output))    
# print(X_test[0].dot(model.coef_))

print(sum(output) + model.intercept_) 
delta = []
print(len(y_pred), len(y_test))
for i in range(len(y_pred)):
    delta.append(y_test[i] - y_pred[i])
# y_pred = sorted(y_pred)
# y_test = sorted(y_test)
print(type(X_test))

X_features = np.transpose(X_test)

# plt.plot(range(0,89), delta, color='blue', linewidth=0.3)
# plt.plot(range(0,89), y_pred, color='blue', linewidth=0.3)
# plt.plot(range(0,89), y_test, color='orange', linewidth=0.3)
# plt.plot(X_features[7], y_pred, '.', color='orange')
# plt.plot(X_features[7], y_test, '.', color='blue')
# plt.plot( y_test)
# plt.plot( y_pred)

# plot a line, a perfit predict would all fall on this line
# x = np.linspace(0, 330, 100)
# print(x)
# y = x * 2
# plt.plot(x, y)

for f in range(0,10):
    plt.scatter(X_features[f], y_test,  color='black')
    plt.scatter(X_features[f], y_pred, color='orange')
    plt.show()
      