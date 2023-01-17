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
#Import the necessary packages



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_squared_error, mean_absolute_error



from sklearn import preprocessing

from sklearn.neighbors import KNeighborsRegressor

import seaborn as sns
df = pd.read_csv("/kaggle/input/board-games-prediction-data/games.csv")

df.head()
df.info()
df = df.drop(columns=["id", "bayes_average_rating", "type", "name"])

df.head()
print(df.shape)
plt.hist(df["average_rating"])
df1 = df[df["average_rating"]==0]

df1.shape
# eliminating all the rows having average_rating = 0 or less than 0, since average rating can not be less than 1.

df = df[df["average_rating"]>0]

df.shape
#Selecting games for which number of users who rated the game is greater than zero.

df = df[df["users_rated"] > 0]

df = df.dropna(axis=0)

df.shape
dfx = df.drop(columns=["average_rating"])

dfy = df["average_rating"]



for i in range(len(dfx.columns)):

    x = dfx.iloc[:,i].values

    y = dfy.values

    plt.figure(figsize=(18,16))

    plt.scatter(x, y)

    plt.xlabel(dfx.columns[i])

    plt.ylabel("average_rating")

    plt.show()
df = df[df["yearpublished"]>1800]

df = df[df["minplayers"]<15]

df = df[df["maxplayers"]<100]

df = df[df["playingtime"]<500]

df = df[df["minage"]<20]

df = df[df["minplaytime"]<200]

df = df[df["maxplaytime"]<200]

print(df.shape)
dfx2 = df.drop(columns=["average_rating"])

dfy2 = df["average_rating"]



for i in range(len(dfx2.columns)):

    x2 = dfx2.iloc[:,i].values

    y2 = dfy2.values

    plt.figure(figsize=(18,16))

    plt.scatter(x2, y2)

    plt.xlabel(dfx2.columns[i])

    plt.ylabel("average_rating")

    plt.show()
correlation_matrix = df.corr()

fig = plt.figure(figsize = (12, 9))

sns.heatmap(correlation_matrix, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
X = df.drop(columns=['average_rating']).values

Y = df['average_rating'].values



train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=4)



model_LR = LinearRegression().fit(train_x, train_y)

model_RFR = RandomForestRegressor(n_estimators = 50, max_depth = 6).fit(train_x, train_y)



y_true = test_y



y_pred_LR = model_LR.predict(test_x)

y_pred_RFR= model_RFR.predict(test_x)



MSE_LR = mean_squared_error(y_true, y_pred_LR)

MSE_RFR = mean_squared_error(y_true, y_pred_RFR)



MAE_LR = mean_absolute_error(y_true, y_pred_LR)

MAE_RFR = mean_absolute_error(y_true, y_pred_RFR)



print("Mean Squared Error for Linear Regressor", "%.2f" % MSE_LR)

print("Mean absolute Error for Linear Regressor", "%.2f" %MAE_LR)



print("Mean Squared Error for Random Forest Regressor", "%.2f" % MSE_RFR)

print("Mean absolute Error for Random Forest Regressor", "%.2f" %MAE_RFR)
X_KNN = X

Y_KNN = Y



from sklearn import preprocessing

X_KNN = preprocessing.normalize(X)



train_xKNN, test_xKNN, train_yKNN, test_yKNN = train_test_split(X_KNN, Y_KNN, test_size=0.2, random_state=4)



y_true_KNN = test_yKNN



MSE_array = []

MSE_min = 10^12

k_best = 5

k_array = []

from sklearn.neighbors import KNeighborsRegressor

for k in range(5,200):

    model = KNeighborsRegressor(n_neighbors=k) 

    model.fit(train_xKNN, train_yKNN)

    y_pred = model.predict(test_xKNN)

    MSE_temp = mean_squared_error(y_true_KNN, y_pred)

    

    if MSE_temp < MSE_min:

        MSE_min = MSE_temp

        k_best = k

    MSE_array.append(MSE_temp)

    k_array.append(k)



print("Best k parameter is ", k_best)
plt.plot(k_array, MSE_array)

plt.show()
model_KNN = KNeighborsRegressor(n_neighbors=k_best)

model_KNN.fit(train_xKNN, train_yKNN)

y_pred_KNN = model_KNN.predict(test_xKNN)



MSE_KNN = mean_squared_error(y_true_KNN, y_pred_KNN)

MAE_KNN = mean_absolute_error(y_true_KNN, y_pred_KNN)

print("Mean Squared Error for KNN Regressor", "%.2f" % MSE_KNN)

print("Mean absolute Error for KNN Regressor", "%.2f" %MAE_KNN)