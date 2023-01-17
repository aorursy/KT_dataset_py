# importing basic libaries for the basic operations

import numpy as np # for the mathematical calculations

import pandas as pd # for reading our data/ csv files

import seaborn as sns # for the visualising the data

from sklearn import metrics # for computing some parameters of our linear regression model

import matplotlib.pyplot as plt # for some pollting of the data points

from sklearn.model_selection import train_test_split # to split our dataset 

from sklearn.linear_model import LinearRegression # the model from sklearn 

df = pd.read_csv("/kaggle/input/startup-logistic-regression/50_Startups.csv")

df
df['State'] = pd.get_dummies(df['State'], prefix='State')
sns.heatmap(df.corr(),annot=True)
# now here using a poweful tool called Slicing we are giving the labels which help in prediction in x and one to predict in y

x = df.iloc[:,:4].values

y = df.iloc[:,4:5].values
# now splitting our dataset as per your choice generally a good pratice means (80:20 or 75:25)

(trainX, testX, trainY, testY) = train_test_split(x, y, test_size=0.25, random_state=0)



print("X Train shape:", trainX.shape)

print("X Test shape:", testX.shape)

print("Y Train shape:", trainY.shape)

print("Y Test shape:", testY.shape)
# training our model a very simple and nice model :)

regressor = LinearRegression()  

regressor.fit(trainX, trainY)
print(regressor.intercept_)

print(regressor.coef_)
# now doing prediction on our model

y_pred = regressor.predict(testX)

y_pred.shape
df = pd.DataFrame({'Actual': testY.flatten(), 'Predicted': y_pred.flatten()})

df
# the bar plot is very helpful when you deal with such numbers this will add a cherry on your cake guys.... and make it easy for everyone to understand !!!!

df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
# and last some parameters to check your model correctly and i am not writting much about these i think just copy and pasting from google is waste so you guys can go and

#search and i am sure you will get much good than me and tell me that wheather my model is good or not based on these parameter values....



print('Mean Absolute Error:', metrics.mean_absolute_error(testY, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(testY, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY, y_pred)))