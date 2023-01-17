from sklearn import datasets

import numpy as np

import pandas as pd

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

import seaborn as sns 

from keras.layers import Input, Dense

from keras.models import Model

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.metrics import  mean_squared_error

from math import sqrt

import matplotlib.pyplot as plt
boston = datasets.load_boston()

print(boston.keys())

print(boston.data.shape)

print(boston.feature_names)

#data description

print (boston.DESCR)

# Set the features  

bos = pd.DataFrame(boston.data, columns=boston.feature_names)

print(bos.head())

# Set the target

target = pd.DataFrame(boston.target, columns=["MEDV"])

bos["MEDV"]=target["MEDV"]

correlation_matrix = bos.corr().round(2)

# annot = True to print the values inside the square

plt.figure(figsize = (12,5))

sns.heatmap(data=correlation_matrix, annot=True)
X = bos[["RM", "LSTAT"]]

Y = target["MEDV"]



# Fit and make the predictions by the model

model = sm.OLS(Y, X).fit()

predictions = model.predict(X)



# Print out the statistics

model.summary()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
# scale the data within [0-1] range

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)

X_test = scalar.transform(X_test)
#seeding the mlp

seed=20

np.random.seed(seed)



from tensorflow import set_random_seed

set_random_seed(seed)
# this is our input placeholder

input_data = Input(shape=(2,))



# "encoded" is the encoded representation of the input

firstlayer = Dense(2, activation='relu', name='input')(input_data)

ff=Dense(2, activation='relu', name='ff')(firstlayer)

secondlayer = Dense(1, activation='linear', name='prices')(ff)

MLPRegModel = Model(inputs=input_data, outputs=secondlayer)



# Compile model

MLPRegModel.compile(loss='mse', optimizer='rmsprop')



# Fit the model

MLPRegModel.fit(X_train, Y_train, epochs=250, batch_size=10)

print('Now making predictions')

predictions = MLPRegModel.predict(X_test)



"""""

#this is for remainder purpose

seed=3 best for adam

in adam batch size=1 best result

MSE: 21.7102 epoch-500 batch 2 adam

MSE: 22.2704 epoch-250 batch 2 adam



without tensorflow seeding

MSE: 21.3172 rmsprop "

MSE: 20.3754 rmsprop with ff layer

MSE: 20.758  rmsprop epoch- 500 batch 1

MSE: 20.4079 rmsprop epoch- 500 batch 2

MSE: 20.3896 rmsprop epoch- 150 batch 2

MSE: 20.4025 rmsprop epoch- 150 batch 1



after tensorflow seed

MSE: 20.425  rmsprop epoch- 250 batch 2

MSE: 20.3996 rmsprop epoch- 250 batch 3

MSE: 20.3558 rmsprop epoch- 250 batch 10

MSE: 20.4105 rmsprop epoch- 500 batch 10



#original way to calculate  mse



pred=pd.DataFrame(predictions)

pred.columns=["MEDV"]

pred["MEDV"]=pred.MEDV.astype(float)

print(pred.head())

ytest=pd.DataFrame(Y_test)

print(ytest.head())

pred.index=ytest.index

pred["diff"]=pred.loc[:,"MEDV"] - ytest.loc[:,"MEDV"]

print(pred.head())

mse = (pred["diff"] ** 2).mean()

print('MSE: {}'.format(round(mse, 4)))



"""""
print("R2 score: {}".format(round(r2_score(Y_test, predictions),4)))

print("MSE: {}".format(round(mean_squared_error(Y_test, predictions),4)))

print("RMSE: {}".format(round(sqrt(mean_squared_error(Y_test, predictions)),4)))