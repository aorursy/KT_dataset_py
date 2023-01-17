#In this analysis we try to decide whether a company should focus their efforts on their mobile app experience or website



print ('Importing libraries')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
#loading and exploring the dataset 



#Loading the dataset

ecomm = pd.read_csv('../input/Ecommerce Customers.csv')  



#exploring the dataset head, info and stats

ecomm.head() 

ecomm.info() 

ecomm.describe()  
#exploratory data analysis (EDA)



#using a regression plot to visualise Time on Website and Yearly Amount Spent corrlation

sns.regplot(x='Time on Website',y='Yearly Amount Spent',data=ecomm,color='gray')  
#using pairplot on the whole datast (numarical values only) in order to look for relationships

sns.pairplot(ecomm)



#Results have shown relationships between Yearly Amount Spent and Length of Membership
#plotting linear model plot in order to see corrlations (plot is presenting positive linear correlation)')

sns.set_style(style='whitegrid')

sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=ecomm)
#training and testing the dataset

ecomm.columns

#assigning all numarical features of the customers to X (independent variables)

X = ecomm[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]

#assigning Yearly Amount Spent to y (dependent variable)

y = ecomm['Yearly Amount Spent']
#importing train_test_split in order to split the dataset 

from sklearn.model_selection import train_test_split
#testsize was set to 0.3 (30% of the dataset) with 101 random state

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#training the model



#importing te Linear Regression model

from sklearn.linear_model import LinearRegression

#creating an object that will contain the training model 

lm = LinearRegression()

#training the model 

lm.fit(X_train,y_train)
#saving and presetning the coefficients of the model

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

print (coeff_df)
#predicting using test dataset

predictions = lm.predict(X_test)
#creating a scatterplot of  real test values vs. predicted values

sns.set_style(style='whitegrid')

sns.scatterplot(y_test,predictions)

plt.xlabel('Y Test - True Values')

plt.ylabel('Predicted Values')
#evaluating the model



#importing metrics from sklearn

from sklearn import metrics
#calculating and printing the errors = MAE, MSE, RMSE

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

metrics.explained_variance_score(y_test,predictions)
#residuals

#plotting the residuals in order to see if they normally distributed

sns.distplot((y_test-predictions),bins=50)
#Conclusion: the Time on App coefficient is 38.59 while Time on Website is only 0.19. 

#This means that people prefer to use the mobile app to place orders. 

#The company can decide to improve their website in order to increase web purchases

#or to improve more the app in order to maximise app based sales.
