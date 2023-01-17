# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np
#input layer'daki 2 node'un value'ları 2 ve 3

input_data = np.array([2,3])
print(input_data)
#hidden layer'daki 2 node'la olan weight'leri 1,1 ve -1,1

weights = {'node_0': np.array([1,1]),

           'node_1': np.array([-1,1]),

           'output': np.array([2,-1])}
print(weights)
#hidden layer'daki node'ların value'ları hesaplanıyor:

# 2*1+3*1=5,,, 2*-1+3*1=1 #bunlar hidden layer'daki iki node'un value'ları

node_0_value = (input_data * weights['node_0']).sum()

node_1_value = (input_data * weights['node_1']).sum()
print(node_0_value)

print(node_1_value)
#hidden layer'daki node'ların value'larını array'e çevirdik:

hidden_layer_values = np.array([node_0_value, node_1_value])

print(hidden_layer_values)
#son olarak hidden layer values'i (2 node) optput weight'leriyle çarparak output'u bulduk:

# 5 * 2 + 1 * -1 = 9

output = (hidden_layer_values * weights['output']).sum()

print(output)
#tanh örneği:



import numpy as np



#input layer'daki 2 node'un value'ları 2 ve 3

input_data = np.array([2,3])



#hidden layer'daki 2 node'la olan weight'leri 1,1 ve -1,1

weights = {'node_0': np.array([1,1]),

           'node_1': np.array([-1,1]),

           'output': np.array([2,-1])}



#hidden layer'daki node'ların input value'ları hesaplanıyor:

# 2*1+3*1=5

node_0_input = (input_data * weights['node_0']).sum()

#ardından bulunan değere tanh activation function uygulanarak hidden node'dan çıkacak değer belirleniyor:

node_0_output = np.tanh(node_0_input)

#2*-1+3*1=1

node_1_input = (input_data * weights['node_1']).sum()

#ardından bulunan değere tanh activation function uygulanarak hidden node'dan çıkacak output değer belirleniyor:

node_1_output = np.tanh(node_1_input)



#hidden layer output value'larını array'e çevirdik:

hidden_layer_outputs = np.array([node_0_output, node_1_output])



#son olarak hidden layer output values'i (2 node) optput weight'leriyle çarparak output'u bulduk:

output = (hidden_layer_outputs * weights['output']).sum()

print(output)
#2 input'lu örneği:



import numpy as np



#input layer weights

weights = np.array([1,2])



#input layer values

input_data = np.array([3,4])



#actual value

target = 6



#learning rate

learning_rate = 0.01



#prediction (output)

preds = (weights * input_data).sum()

print(preds)

#error

error = preds - target

print(error)
#slope calculation:



#gradient yukarıdaki -24 olarak hesapladığımız değerdi, burada da:

gradient = 2 * input_data * error

print(gradient)



#updated weight ilk weight'ten gradient*learning_rate kadar çıkarınca elde edilen

weights_updated = weights - learning_rate * gradient

print(weights_updated)



#weight yerine weight_updated kullandık

preds_updated = (weights_updated * input_data).sum()

print(preds_updated)



error_updated = preds_updated - target

print(error_updated)
#import what we will need

import numpy as np #for reading data

from keras.layers import Dense #these two imports are used for building your model

from keras.models import Sequential
#these lines are for reading data

predictors = pd.read_csv('../input/hourly_wages.csv')

n_cols = predictors.shape[1] #we read the data so we can find the number of nodes in the input layer which is stored as n_cols

#we always need to specify how many columns are in the input when building a Keras model, because that is the number of nodes in the input layer
target = predictors['wage_per_hour'].values
#we start the model:

model = Sequential() #Sequential is one of two ways of building a model which is the easier way to build a model.

#Sequential models require that each layer has weights or connections only to the one layer coming directly after it in the network diagram.

#we add layers using the add metod of the model. type of layer you see is the standard layer type, is called Dense layer.

#Its called dense because all the nodes in the previous layer connect to all of the nodes in the current layer.

model.add(Dense(100, activation='relu', input_shape= (n_cols,))) #In each layer we specify the number of nodes as the first positional argument (100-the value is common to use), 

#and the activation function you want to use.

#In the first layer we NEED to specify input shapes as shown: input_shape= (n_cols,). That says the input will have n_cols columns, and there is nothing after the comma, 

#meaning it can have any number of rows, that is any number of data points.

model.add(Dense(100, activation='relu')) 

model.add(Dense(1)) #last layer has one node that is the output or prediction of the model.



#This model has 2 hidden layers and an output layer.
# Compile the model:

#two important arguments: 

#1- what optimizer to use, which controls the learning rate ('adam' is the best choice for learning rate)

#2- loss function (mean_squared_error is the most common choice for regression)

model.compile(optimizer='adam', loss='mean_squared_error')



# Verify that model contains information from compiling

print("Loss function: " + model.loss)
#Fitting the model: applying backpropagation and gradient descent with your data to update the weights.

#scaling the data before fitting can ease optimization (sonra yap)

model.fit(predictors, target)
#import what we will need

import numpy as np #for reading data

from keras.layers import Dense #these two imports are used for building your model

from keras.models import Sequential

from keras.utils import to_categorical #convert the data from one column to multiple columns 

#import data

data = pd.read_csv('../input/titanic/titanic_all_numeric.csv')
data.head()
#predictor variable data:

predictors = data.drop(['survived'], axis=1).as_matrix()



n_cols = predictors.shape[1]
#target data: (categorical yapıldı)

target = to_categorical(data.survived)
#we start the model:

model = Sequential()
#we add layers

model.add(Dense(100, activation='relu', input_shape= (n_cols,))) 

model.add(Dense(100, activation='relu')) 

model.add(Dense(100, activation='relu')) 

model.add(Dense(2, activation='softmax')) #that is the difference mentioned in 2
# Compile the model:

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy']) #that is the difference mentioned in 1
#Fitting the model: 

model.fit(predictors, target)



#lets look at results:

#both accuracy and loss improve for the first 3 epochs, and then improvement slows down. 
#here is the code for saving, reloading and making predictions:

#import load model function

from keras.models import load_model

#save

model.save('model_file.h5') #h5 is the common extension

#load

my_model = load_model('model_file.h5')

#predict

predictions = my_model.predict(predictors) #argument is data_to_predict_with, bağımsız bir data set, predict edilecek



probability_true = predictions[:,1] #second column is the 1 values so 1st column'ı alıyoruz
data = pd.read_csv('../input/titanic/titanic_all_numeric.csv')



predictors = data.drop(['survived'], axis=1).as_matrix()



n_cols = predictors.shape[1]



input_shape= (n_cols,)



from keras.utils import to_categorical



target = to_categorical(data.survived)
import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD





#we have a function that creates a new model

def get_new_model(input_shape = input_shape):

    model = Sequential()

    model.add(Dense(100, activation='relu', input_shape = input_shape))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(2))

    return model



#learning rates:

lr_to_test = [0.000001, 0.01, 1]





#we create models in a for loop with different lrs

for lr in lr_to_test:

    model = get_new_model()

    my_optimizer = SGD(lr=lr)

    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

    model.fit(predictors, target)
#import data and prepare:



data = pd.read_csv('../input/titanic/titanic_all_numeric.csv')



predictors = data.drop(['survived'], axis=1).as_matrix()



n_cols = predictors.shape[1]



input_shape= (n_cols,)



from keras.utils import to_categorical



target = to_categorical(data.survived)
#import functions:

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD
#Instantiate model:

model = Sequential()

model.add(Dense(100, activation='relu', input_shape = input_shape))

model.add(Dense(100, activation='relu'))

model.add(Dense(2))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy']) #classification olduğu için metrics = ['accuracy']

model.fit(predictors, target, validation_split=0.3) #validation_split=0.3 ile %30 unu validation için ayırdık.
#import function

from keras.callbacks import EarlyStopping
#create early stopping monitor before fitting model

early_stopping_monitor = EarlyStopping(patience=2) #how many epochs the model can go without improving before we stop training.



fitted = model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks=[early_stopping_monitor]) #epochs=1-->>default



#son 2 epoch'da val_loss value 0.5532'yi geçemediği için 6da durdu.



# Create the plot

plt.plot(fitted.history['val_loss'], 'r')

plt.xlabel('Epochs')

plt.ylabel('Validation score')

plt.show()