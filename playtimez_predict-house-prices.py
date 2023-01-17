import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import matplotlib.pyplot as plt 
import math 
data = pd.read_csv("../input/kc_house_data.csv")
data.head()
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors']]
Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 1/3, random_state = 0)
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
plt.scatter(data['sqft_living'], Y)
data['sqft_living'].hist(bins = 50)
model = LinearRegression()
model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
pred_train = model.predict(xtrain)
mse_train = ((pred_train - ytrain)**2).sum()/len(ytrain)
mse_train
(abs(pred_train - ytrain)/ytrain).sum()/len(ytrain)
predtest = model.predict(xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)
