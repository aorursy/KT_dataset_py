# just import

import numpy as np 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

# from tensorflow import keras

import joblib

from sklearn.neural_network import MLPRegressor

import random as random

import numpy as np 

from sklearn.preprocessing import MinMaxScaler

TRAIN = True
# create data



def func(x):

     return np.cos(x)



x = np.linspace(-10,10,400)

x_post = np.linspace(10, 20, 200)



y = func(x)

y_post = func(x_post)



# plot main function

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.spines['left'].set_position('center')

ax.spines['bottom'].set_position('zero')

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')



plt.plot(x,y, 'r')

plt.title("y(x)")

plt.show()
# split data

x_train, x_test, y_train, y_test = train_test_split(x, y, 

                                test_size=0.25, shuffle=True)





x_train = x_train.reshape(-1, 1)

x_test = x_test.reshape(-1, 1)

y_train = y_train.reshape(-1, 1).ravel()

y_test = y_test.reshape(-1, 1)





# normslization 

min_max_scaler = MinMaxScaler(feature_range = (-1, 1))

x_train = min_max_scaler.fit_transform(x_train)

x_test= min_max_scaler.transform(x_test)

x_post_transformed=min_max_scaler.transform(x_post.reshape(-1, 1))





# show first 5 samples of train date

from tabulate import tabulate

table = [

    ["x_train", "y_train"],

     [x_train[:5],y_train[:5]]

]

print(tabulate(table))

if TRAIN:

    

    model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100 ), activation = "relu",

                          max_iter = 1000, verbose=True)

    model.fit(x_train, y_train)

    

else:

    #Loading trained model

    model = joblib.load('_1Version.pkl')



y_pred = model.predict(x_test)

y_pred_post = model.predict(x_post_transformed.reshape(-1, 1))

plt.plot(x_post,y_post, 'r', label = 'cos(x)')

plt.plot(x_post,y_pred_post, 'g', label = 'models prediction')

plt.title("Post Test Part")

plt.legend()

plt.show()