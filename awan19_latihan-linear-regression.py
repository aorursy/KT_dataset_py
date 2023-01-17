## Requirements:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')
## import the car dataset

df = pd.read_csv('../input/car_data.csv') # Importing the dataset

df.sample(5) #previewing dataset randomly
## print 5 sample dataset values

print(df.shape) # view the dataset shape

print(df['Make'].value_counts()) # viewing Car companies with their cars number
## print the shape of the dataset and print the different car companies with their total cars

new_df = df[df['Make']=='Volkswagen'] # in this new dataset we only take 'Volkswagen' Cars

print(new_df.shape) # Viewing the new dataset shape

print(new_df.isnull().sum()) # Is there any Null or Empty cell presents

new_df = new_df.dropna() # Deleting the rows which have Empty cells
## view the shape and check if any null cell present or not

print(new_df.shape) # After deletion Vewing the shape

print(new_df.isnull().sum()) #Is there any Null or Empty cell presents

new_df.sample(2) # Checking the random dataset sample
## select only 2 specific (‘Engine HP’ and ‘MSRP’) columns from all columns

new_df = new_df[['Engine HP','MSRP']] # We only take the 'Engine HP' and 'MSRP' columns

new_df.sample(5) # Checking the random dataset sample
## put the ‘Engine HP’ column as a numpy array into ‘X’ variable. 

## ‘MSRP’ column as a numpy array into ‘y’ variable. 

## Then check the shape of the array.

X = np.array(new_df[['Engine HP']]) # Storing into X the 'Engine HP' as np.array

y = np.array(new_df[['MSRP']]) # Storing into y the 'MSRP' as np.array

print(X.shape) # Vewing the shape of X

print(y.shape) # Vewing the shape of y
## perform a linear regression for prediction.

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=15) # Spliting into train & test dataset

regressor = LinearRegression() # Creating a regressior

regressor.fit(X_train,y_train) # Fiting the dataset into the model
## plot a scatter plot graph between X_test and y_test datasets and we draw a regression line

plt.scatter(X_test,y_test,color="green") # Plot a graph with X_test vs y_test

plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3) # Regressior line showing

plt.title('Regression(Test Set)')

plt.xlabel('HP')

plt.ylabel('MSRP')

plt.show()
## plot the final X_train vs y_train scatterplot graph with a best-fit regression line

plt.scatter(X_train,y_train,color="blue")  # Plot a graph with X_train vs y_train

plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3) # Regressior line showing

plt.title('Regression(training Set)')

plt.xlabel('HP')

plt.ylabel('MSRP')

plt.show()
## make prediction

y_pred = regressor.predict(X_test)

print('R2 score: %.2f' % r2_score(y_test,y_pred)) # Priniting R2 Score

print('Mean squared Error :',mean_squared_error(y_test,y_pred)) # Priniting the mean error



def car_price(hp): # A function to predict the price according to Horsepower

    result = regressor.predict(np.array(hp).reshape(1, -1))

    return(result[0,0])
## prediction result

car_hp = int(input('Enter Volkswagen cars Horse Power : '))

print('This Volkswagen Prce will be : ',int(car_price(car_hp))*69)