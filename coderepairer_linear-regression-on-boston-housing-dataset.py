%matplotlib inline



#Import Pandas, Numpy, Seaborn and Pyplot libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#Load Boston data

from sklearn.datasets import load_boston

boston = load_boston()



print(boston.DESCR)

print(boston.keys())
#type(boston.data) => numpy.ndarray



#Convert Boston Numpy Array data to DataFrame and add target values n=in same dataframe

boston_data = pd.DataFrame(boston.data,columns=boston.feature_names)

boston_data['MEDV'] = boston.target

#View first 5 rows of data

boston_data.head()
#View last 5 rows of data

boston_data.tail()
#Check the count of null values in dataframe

boston_data.isnull().sum()



#Inference: Target value is normally distrubuted with very few outliers
# Plot distribution graph for target value 'MEDV'

sns.set(rc={'figure.figsize':(12,7)})

sns.distplot(boston_data['MEDV'], bins=30)

plt.show()
#Plot heatmap for all correlations in the data

sns.heatmap(data=boston_data.corr(),annot=True)



#It is observed that target MEDV is highly correlated with LSTAT and RM features. 

#Also it is observed some of the features are correlated with each other for e.g.(AGE & DS), (RAD & tax) 

# Inter correlated features are not considered because it is diffult to infer target value is correlated with which of the one. 
# visualize the relationship between the selected features(RM, LSTAT) and the response(MEDV) using scatterplots

sns.pairplot(boston_data, x_vars=['RM','LSTAT'], y_vars='MEDV', size=7, aspect=0.7, kind='reg')



#It is observed that MEDV has linear relationship between these 2 features. 

# MEDV & RM has positive linear relationship 

# MEDV & LM has negative linear relationship
#Model is built using Linear Regression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



# Features selected

features = ['RM','LSTAT']



x = boston_data[features]

y = boston_data['MEDV']



#Split the the data in 80/20 ratio

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True)



#Print shapes

print("X_train shape : ",X_train.shape)

print("X_test shape: ",X_test.shape)

print("y_train shape: ",y_train.shape)

print("y_test shape: ",y_test.shape)



linreg = LinearRegression()

linreg.fit(X_train,y_train)



#Print Intercept value

print("Intercept :: ", linreg.intercept_)



#Print Linear Coefficient for each feature

print("Coefficients :: ",list(zip(features,linreg.coef_)))
#Calculate Root Mean Square Error



y_predict = linreg.predict(X_test)

print(np.sqrt(metrics.mean_absolute_error(y_test, y_predict)))
