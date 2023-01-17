import numpy as np 

import pandas as pd 

import scipy

from scipy.stats import pearsonr

from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split 

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from math import sqrt

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn import datasets

from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

from yellowbrick.regressor import ResidualsPlot

%matplotlib inline
np.random.seed(1)

X1 = np.random.randint(500, 2000, 50)

X1
np.random.seed(1)

X2 = np.random.randint(100, 500, 50)

X2
X3 = [i * 3 + np.random.randint(500) for i in X1]

X3
Y = [6 * X1[i] + X2[i] for i in range(0,50)]

Y
data = {'X1':X1, 'X2':X2, 'X3': X3, 'Y': Y} 

df = pd.DataFrame(data) 

df
#using pandas

corr = df['X1'].corr(df['Y'])

corr
#Spearmans's 's correlation coefficient

corr, _ = spearmanr(X1, Y)

corr
# Pearson's correlation coefficient

corr, _ = pearsonr(X1, Y)

corr
#using pandas

corr = df['X2'].corr(df['Y'])

corr
#Spearmans's correlation coefficient

corr, _ = spearmanr(X2, Y)

corr

#Spearmans's 's correlation coefficient

corr, _ = spearmanr(X2, Y)

corr
#using pandas

corr = df['X3'].corr(df['Y'])

corr
# Pearson's correlation coefficient

corr, _ = pearsonr(X3, Y)

corr
#Spearmans's 's correlation coefficient

corr, _ = spearmanr(X3, Y)

corr
plt.scatter(X1, Y, alpha=0.5)

plt.title('Scatter plot illustrating the relationship between X1 and Y')

plt.xlabel('X1')

plt.ylabel('Y')

plt.ylim(bottom=0)

plt.xlim(left=0)

plt.show()
plt.scatter(X2, Y, alpha=0.5)

plt.title('Scatter plot illustrating the relationship between X2 and Y')

plt.xlabel('X2')

plt.ylabel('Y')

plt.ylim(bottom=0)

plt.xlim(left=0)

plt.show()
# Separate our data into independent(X) variables.

X_data = df[['X1','X2']]

X_data
# Separate our data into dependent(Y) variables.

Y_data = df['Y']

Y_data
# 70/30 Train Test Split.

# We will split the data using a 70/30 split. i.e. 70% of the data will be randomly 

# chosen to train the model and 30% will be used to evaluate the model

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
# Create an instance of linear regression

reg = linear_model.LinearRegression()
# Fitting the X_train onto y_train.

reg.fit(X_train,y_train)
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
# Intercept

reg.intercept_
# Make predictions using the testing set

test_predicted = reg.predict(X_test)

test_predicted
# Explained variance score: 1 is perfect prediction

# R squared

print('Variance score: %.2f' % r2_score(y_test, test_predicted))
reg.score(X_test,y_test)
scores = cross_val_score(reg,X_data, Y_data, cv=5)

scores     
#Residual Plot

plt.scatter(reg.predict(X_train), reg.predict(X_train)-y_train,c='b',s=40,alpha=0.5)

plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,c='g',s=40)

plt.hlines(y=0,xmin=np.min(reg.predict(X_test)),xmax=np.max(reg.predict(X_test)),color='red',linewidth=3)

plt.title('Residual Plot using Training (blue) and test (green) data ')

plt.ylabel('Residuals')
model = Ridge()

visualizer = ResidualsPlot(model)



visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)

visualizer.score(X_test, y_test)



visualizer.show()
x=342

y=21

data = {'x':[x],'y':[y]}

df=pd.DataFrame(data)

reg.predict(df)
#MAE

mean_squared_error(y_test, test_predicted)
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))

print("Mean Absolute error: %.2f" % mean_absolute_error(y_test, test_predicted))

print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, test_predicted)))