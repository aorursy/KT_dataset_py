import os

####*IMPORANT*: Have to do this line *before* importing tensorflow

os.environ['PYTHONHASHSEED']=str(1)



import tensorflow as tf

import tensorflow.keras as keras

import tensorflow.keras.layers

import random

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
def reset_random_seeds():

    os.environ['PYTHONHASHSEED']=str(1)

    tf.random.set_seed(1)

    np.random.seed(1)

    random.seed(1)
#synthetic data function

num_features = 10

num_samples = 30000

def synth_func(X):

    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose()



    f1 = np.exp(np.abs(X1-X2))                        

    f2 = np.abs(X2*X3)  

    f3 = -1*(X3**2)**np.abs(X4)

    f4 = (X1*X4)**2

    f5 = np.log(X4**2 + X5**2 + X7**2 + X8**2)

    f6 = X9 + 1/(1 + X10**2)



    Y =     f1 + f2 + f3 + f4 + f5 + f6    

    return Y
#generate synthetic data

np.random.seed(1)

X = np.random.uniform(low=-1, high=1, size=(num_samples,num_features))

data = pd.DataFrame(X)

data = data.rename({0:"x1",1:"x2",2:"x3",3:"x4",4:"x5",5:"x6",6:"x7",7:"x8",8:"x9",9:"x10"},axis=1)

Y = synth_func(X)

data["y"]=Y

data.head()
#test_train_split & data Scaling

X = data.iloc[:,0:10].values

y = data["y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
#model building

def train_model(X_train, y_train):

    reset_random_seeds()

    model = keras.Sequential([

            keras.layers.Dense(40, input_dim=X_train.shape[1], activation='relu'),

            keras.layers.Dense(20, activation='relu'),

            keras.layers.Dense(10, activation='relu'),

            keras.layers.Dense(1, activation='relu')

        ])



    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_absolute_error'])

    #keeping epochs = 1 for the reproducability test

    

    

    model.fit(X_train, y_train, epochs=2,batch_size=100)
#for each run NN results are reproducable

print("train_model - Run : 1")

train_model(X_train, y_train)

print("")

print("")

print("train_model - Run : 2")

train_model(X_train, y_train)