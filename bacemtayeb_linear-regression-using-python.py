# Data Loading + Cleaning
import pandas as pd
# Some Math
import numpy as np
# Visualization 
import matplotlib.pyplot as plt
# Linear Model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
# We start by loading the training data
train_data = pd.read_csv('../input/train.csv')
train_data = pd.DataFrame(train_data)
# And also the testing data
testing_data = pd.read_csv('../input/test.csv')
testing_data = pd.DataFrame(testing_data)

# Let's get some information
info = pd.Series([np.round(train_data.x.mean(),1),train_data.y.isna().sum(),train_data.size],index=['Mean','NaN Values','Size'])
info
#Let's do some cleaning and look again
clean_train_data = train_data.dropna()
clean_train_data.size
# Linear Regression requires visualization for better exploration
plt.scatter(x = clean_train_data.x, y = clean_train_data.y)
#Super ! Through the previous plot our data sounds to be perfectly correlated!
clean_train_data.corr(method='pearson')
# 0.99 is our corr coef and this is more than amazing! Let's get into our model
# Convert our values to matrix
x_train = clean_train_data.as_matrix(['x'])
y_train = clean_train_data.as_matrix(['y'])
#Let's load our model and fit the data
lm = linear_model.LinearRegression()
lm.fit(x_train,y_train)
clean_test = testing_data.dropna()
test_x = clean_test.as_matrix(['x'])
test_y= clean_test.as_matrix(['y'])
pred = lm.predict(test_x)
print('Coeff :',lm.coef_)
print('MSE : ',mean_squared_error(test_y,pred))

