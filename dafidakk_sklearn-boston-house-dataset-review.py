# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from pandas import Series,DataFrame




import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.datasets import load_boston




# Load the housing dataset

boston = load_boston()







print(boston.DESCR)







# Histogram of prices (this is the target of our dataset)

plt.hist(boston.target,bins=50)



#label

plt.xlabel('Price in $1000s')

plt.ylabel('Number of houses')







# Plot the column at the 5 index (Labeled RM)

plt.scatter(boston.data[:,5],boston.target)



#label

plt.ylabel('Price in $1000s')

plt.xlabel('Number of rooms')



# reset data as pandas DataFrame

boston_df = DataFrame(boston.data)



# label columns

boston_df.columns = boston.feature_names



#show

boston_df.head()




# Set price column for target

boston_df['Price'] = boston.target







# Show result

boston_df.head()







# Using seabron to create a linear fit

sns.lmplot('RM','Price',data = boston_df)



X = boston_df.RM
X.shape
 #Using Numpy for a Univariate Linear Regression

X=np.vstack([boston_df.RM,np.ones(len(boston_df.RM))]).T
X.shape
Y = boston_df.Price
X
url = 'https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html'

"""

m  = Coefficient

b  = bias



"""



# Now get out m and b values for our best fit line

m , b = np.linalg.lstsq(X,Y,rcond=None)[0]

m
b
# First the original points, Price vs Avg Number of Rooms

plt.plot(boston_df.RM,boston_df.Price,'o')



# Next the best fit line

x= boston_df.RM

plt.plot(x, m*x + b,'r',label='Best Fit Line')
# Get the resulting array

result = np.linalg.lstsq(X,Y,rcond=None)



# Get the total error

error_total = result[1]



# Get the root mean square error

rmse = np.sqrt(error_total/len(X) )



# Print

print("The root mean squared error was %.2f " %rmse)
# Import for Linear Regression

import sklearn

from sklearn.linear_model import LinearRegression

# Create LinearRegression object



lreg = LinearRegression()
# Data Columns

X_multi = boston_df.drop('Price',1)



# Targets

Y_target = boston_df.Price
# Implement Linear Regression

lreg.fit(X_multi,Y_target)
print(' The estimated intercept coefficient is %.2f ' %lreg.intercept_)
print(' The number of coefficients used was %d ' % len(lreg.coef_))




# Set a DataFrame from the Features

coeff_df = DataFrame(boston_df.columns)

coeff_df.columns = ['Features']



# Set a new column lining up the coefficients from the linear regression

coeff_df["Coefficient Estimate"] = pd.Series(lreg.coef_)



# Show

coeff_df



X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,boston_df.Price)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
lreg = LinearRegression()



lreg.fit(X_train,Y_train)
pred_train = lreg.predict(X_train)

pred_test = lreg.predict(X_test)
print("Fit a model X_train and calculate the MSE with Y_train: %.2f" %np.mean((Y_train-pred_train)**2))

print("Fit a model X_train and calculate the MSE with X_test and  Y_test: %.2f" %np.mean((Y_test-pred_test)**2))
# Scatter plot the training data

train = plt.scatter(pred_train,(Y_train-pred_train),c='b',alpha=0.5)



# Scatter plot the testing data

test = plt.scatter(pred_test,(Y_test-pred_test),c='r',alpha=0.5)



# Plot a horizontal axis line at 0

plt.hlines(y=0,xmin=-20,xmax=60)



#Labels

plt.legend((train,test),('Training','Test'),loc='upper right')

plt.title('Residual Plots')