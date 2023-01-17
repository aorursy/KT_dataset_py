# Import libraries necessary for this project

import numpy as np

import pandas as pd



#Visualization Libraries

import seaborn as sns

import matplotlib.pyplot as plt



#To plot the graph embedded in the notebook

%matplotlib inline
#imports from sklearn library

from sklearn import datasets

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error
#loading the dataset direclty from sklearn

boston = datasets.load_boston()
print(type(boston))

print('\n')

print(boston.keys())

print('\n')

print(boston.data.shape)

print('\n')

print(boston.feature_names)
print(boston.DESCR)
bos = pd.DataFrame(boston.data, columns = boston.feature_names)

bos['PRICE'] = boston.target



bos.head()
bos.isnull().sum()
bos.describe()
sns.set(rc={'figure.figsize':(11.7,8.27)})

plt.hist(bos['PRICE'], bins=30)

plt.xlabel("House prices in $1000")

plt.show()
#Created a dataframe without the price col, since we need to see the correlation between the variables

bos_1 = pd.DataFrame(boston.data, columns = boston.feature_names)

correlation_matrix = bos_1.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True)
plt.figure(figsize=(20, 5))



features = ['LSTAT', 'RM']

target = bos['PRICE']



for i, col in enumerate(features):

    plt.subplot(1, len(features) , i+1)

    x = bos[col]

    y = target

    plt.scatter(x, y, marker='o')

    plt.title("Variation in House prices")

    plt.xlabel(col)

    plt.ylabel('"House prices in $1000"')
X_rooms = bos.RM

y_price = bos.PRICE





X_rooms = np.array(X_rooms).reshape(-1,1)

y_price = np.array(y_price).reshape(-1,1)



print(X_rooms.shape)

print(y_price.shape)
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_rooms, y_price, test_size = 0.2, random_state=5)



print(X_train_1.shape)

print(X_test_1.shape)

print(Y_train_1.shape)

print(Y_test_1.shape)
reg_1 = LinearRegression()

reg_1.fit(X_train_1, Y_train_1)



y_train_predict_1 = reg_1.predict(X_train_1)

rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))

r2 = round(reg_1.score(X_train_1, Y_train_1),2)



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))

print("\n")
# model evaluation for test set

y_pred_1 = reg_1.predict(X_test_1)

rmse = (np.sqrt(mean_squared_error(Y_test_1, y_pred_1)))

r2 = round(reg_1.score(X_test_1, Y_test_1),2)



print("The model performance for training set")

print("--------------------------------------")

print("Root Mean Squared Error: {}".format(rmse))

print("R^2: {}".format(r2))

print("\n")
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1) 

plt.scatter(X_rooms,y_price)

plt.plot(prediction_space, reg_1.predict(prediction_space), color = 'black', linewidth = 3)

plt.ylabel('value of house/1000($)')

plt.xlabel('number of rooms')

plt.show()
X = bos.drop('PRICE', axis = 1)

y = bos['PRICE']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)



reg_all = LinearRegression()

reg_all.fit(X_train, y_train)



# model evaluation for training set



y_train_predict = reg_all.predict(X_train)

rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))

r2 = round(reg_all.score(X_train, y_train),2)



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))

print("\n")
# model evaluation for test set



y_pred = reg_all.predict(X_test)

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

r2 = round(reg_all.score(X_test, y_test),2)



print("The model performance for training set")

print("--------------------------------------")

print("Root Mean Squared Error: {}".format(rmse))

print("R^2: {}".format(r2))

print("\n")
plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Prices ($1000)")

plt.ylabel("Predicted House Prices: ($1000)")

plt.xticks(range(0, int(max(y_test)),2))

plt.yticks(range(0, int(max(y_test)),2))

plt.title("Actual Prices vs Predicted prices")