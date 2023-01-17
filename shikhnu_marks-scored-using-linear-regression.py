#Importing necessary Libraries:



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Reading data from remote link

url = "http://bit.ly/w-data"

df = pd.read_csv(url)



#To read first 5 enties in the DF

df.head()
df.shape #size of DataFrame
# To find the datatypes and missing values if any

df.info() 
# summary statistics

df.describe().T
# Scatter plot to see the distribution of data

plt.figure(figsize=(10,5))

sns.scatterplot(x=df.Hours,y=df.Scores)

plt.xlabel('Hours Studied')  

plt.ylabel('Marks Scored')  

plt.show()
#correlation plot

plt.figure(figsize=(5,5))

correlation_matrix = df.corr()

# annot = True to print the values inside the square

sns.heatmap(data=correlation_matrix, annot=True)

plt.show()
# Dividing the DF to independent and dependent variable

X = df['Hours'].values.reshape(-1,1)

y = df['Scores']
# Spliting the X,y into train and test 



from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)
# Importing LinearRegression from sklearn

from sklearn.linear_model import LinearRegression



# Creating object and fitting the model

lin_reg = LinearRegression()

model = lin_reg.fit(X_train,y_train)

# Plotting the regression line

best_fitline = model.coef_*X+model.intercept_



# Plotting for the data

plt.scatter(X, y)

plt.plot(X, best_fitline, color = 'r');

plt.show()
# Predicting for test dataset

y_pred = model.predict(X_test)
# Creating Actual and Predicted dataset 

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1
# Model Evaluation 



# Importing metrics from sklearn 

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error



# To find Mean Absolute Error(mse)

mse = (mean_absolute_error(y_test, y_pred))

print("MAE:",mse)



# To find Root Mean Squared Error(rmse)

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

print("RMSE:",rmse)



# To find coefficient of determination

r2 =  r2_score(y_test, y_pred)

print("R-Square:",r2)
# Testing with your own data

hours = np.array([9.25]) # No. of hours should be mentioned inside array

hours = hours.reshape(-1,1)

own_pred = model.predict(hours)

print("No of Hours = {}".format(float(hours)))

print("Predicted Score = {}".format(round(own_pred[0],2)))