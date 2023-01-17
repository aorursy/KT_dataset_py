import numpy as np 

import pandas as pd 

import keras as kr



#Retreive base dirname from Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# save filepath to variable for easier access

data_filepath = '/kaggle/input/titanic/train.csv'

# read the data and store data in DataFrame titled melbourne_data

titanic_data = pd.read_csv(data_filepath) 

# define y and X

y = titanic_data['Survived']

X = titanic_data.drop("Survived", axis = 1)

X = X.drop("Name", axis = 1)

X = X.drop("Sex", axis = 1)

X = X.drop("Ticket", axis = 1)

X = X.drop("Embarked", axis = 1)

X = X.drop("Cabin", axis = 1)



print(X.head())

#print(y.head())



# Does y and x contain any null values ?

#print("y has NaN:", y.isnull().values.any())

#print("X has NaN:", X.isnull().values.any())
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)



print("Done!")
# Get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]



# Drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)



print("Drop!")
def pipline(inp_dim,

            n_nod,

            act_fun = 'relu',

            out_act_fun = 'sigmoid',

            opt_method = 'Adam',

            cost_fun = 'binary_crossentropy',

            lr_rate = 0.01, 

            lambd = 0.0, 

            num_out = None):

    

    lays = [inp_dim] + n_nod

    

    main_input = Input(shape=(inp_dim,), dtype='float32', name='main_input')

    

    X = main_input

    for nod in n_nod:

        X = Dense(nod, 

                  activation = act_fun,

                  kernel_regularizer=regularizers.l2(lambd))(X)

        

    output = Dense(num_out, activation = out_act_fun )(X)

    

    method = getattr(optimizers, opt_method)

    

    model =  Model(inputs=[main_input], outputs=[output])

    model.compile(optimizer = method(lr = lr_rate, clipnorm = 1.0),

                  loss = cost_fun,

                  metrics=['accuracy', 'mse'])   

    

    return model
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from tensorflow.keras import metrics, regularizers, optimizers



# Here we need to normalize the data



# Define the network, cost function and minimization method

INPUT = {'inp_dim': X_train.shape[1],         

         'n_nod': [1],                      # number of nodes in hidden layer

         'act_fun': 'tanh',                 # activation functions for the hidden layer

         'out_act_fun': 'sigmoid',          # output activation function

         'opt_method': 'SGD',               # minimization method

         'cost_fun': 'binary_crossentropy', # error function

         'lr_rate': 0.1,                    # learningrate

         'num_out' : 1 }              # if binary --> 1 |  regression--> num inputs | multi-class--> num of classes



# Get the model

model = pipline(**INPUT)



# Print a summary of the model

model.summary()



# Train the model

estimator = model.fit(X_train, y_train,

                      #epochs = 300,                     # Number of epochs

                      validation_data=(X_valid, y_valid),  # We don't have any validation dataset!

                      batch_size = X_train.shape[0],    # Use batch learning

                      #batch_size=25,                   

                      verbose = 1)







# Some plotting

#plt.plot(estimator.history['loss'])

#plt.title('Model training')

#plt.ylabel('training error')

#plt.xlabel('epoch')

#plt.legend(['train'], loc=0)

#plt.show()
