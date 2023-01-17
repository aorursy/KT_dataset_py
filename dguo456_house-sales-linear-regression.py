import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
data = pd.read_csv("../input/kc_house_data.csv")
data.head()
print(data.shape)
print(data.dtypes)
print(data.describe())
X = data[['bedrooms','bathrooms','sqft_living','floors']]
y = data['price']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
print(Xtrain.shape)
print(Xtest.shape)
print(type(Xtrain))
print(type(ytrain))
Xtrain = np.asmatrix(Xtrain)
Xtest = np.asmatrix(Xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
print(type(Xtrain))
print(type(ytrain))
plt.scatter(X['sqft_living'], y)
plt.show()
X['sqft_living'].hist()
plt.show()
model = LinearRegression()
model.fit(Xtrain, ytrain)
print(pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))))
model.intercept_
pred = model.predict(Xtrain)
((pred-ytrain)*(pred-ytrain)).sum() / len(ytrain)  # MSE
predtest = model.predict(Xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)