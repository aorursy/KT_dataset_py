from IPython.display import Image

Image("../input/image-/1.png")

Image("../input/image-/2.png")
Image("../input/image-/3.png")
Image("../input/image-/4.png")
Image("../input/image-/5.png")
Image("../input/image-/6.png")
Image("../input/image-/7.png")
Image("../input/image-/2layers.png")
Image("../input/image-/4layers.png")

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import timeit
import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.initializers import RandomNormal,glorot_normal
from sklearn.model_selection import StratifiedKFold

from keras import backend as K
from keras import initializers,regularizers,constraints
from keras.engine import Layer
from keras.initializers import RandomNormal,glorot_normal
from keras.layers import Input, Embedding, Dense,concatenate,InputSpec
from keras.layers import  Activation,activations,concatenate,InputSpec
from keras.legacy import interfaces
from keras.utils import conv_utils,np_utils ,plot_model
from keras.utils.generic_utils import func_load,deserialize_keras_object,has_arg,get_custom_objects
from keras.utils.generic_utils import deserialize_keras_object,func_dump
np.random.seed(7)

#Defining New Activation functions
############################################################################
def X_1(x):
    return (K.pow(x,1))
get_custom_objects().update({'X_1': Activation(X_1)})
############################################################################
def X_2(x):
    return (K.pow(x,2))/2
get_custom_objects().update({'X_2': Activation(X_2)})
############################################################################
def X_3(x):
    return (K.pow(x,3))/6
get_custom_objects().update({'X_3': Activation(X_3)})
############################################################################
def X_4(x):
    return (K.pow(x,4))/24
get_custom_objects().update({'X_4': Activation(X_4)})
############################################################################
def X_5(x):
    return (K.pow(x,5))/120
get_custom_objects().update({'X_5': Activation(X_5)})
###############################################################################
def X_6(x):
    return (K.pow(x,6))/720
get_custom_objects().update({'X_6': Activation(X_6)})
############################################################################
def X_7(x):
    return (K.pow(x,7))/5040
get_custom_objects().update({'X_7': Activation(X_7)})
############################################################################
def X_8(x):
    return (K.pow(x,8))/40320
get_custom_objects().update({'X_8': Activation(X_8)})
###############################################################################
def X_9(x):
    return (K.pow(x,8))/362880
get_custom_objects().update({'X_9': Activation(X_9)})
###############################################################################

class Dense_Co(Layer):
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 hidden_dim=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense_Co, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        if hidden_dim!=None :
            self.hidden_dim = hidden_dim
        else :
            self.hidden_dim=self.units
                

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]     
##########################################################################      
        self.kernel = self.add_weight(shape=(input_dim, self.hidden_dim*6),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
    
    
        self.kernel1 = self.kernel[:, :self.hidden_dim]
        self.kernel2 = self.kernel[:, self.hidden_dim: self.hidden_dim * 2]
        self.kernel3 = self.kernel[:, self.hidden_dim * 2: self.hidden_dim * 3]
        self.kernel4 = self.kernel[:, self.hidden_dim * 3: self.hidden_dim * 4]   
        self.kernel5 = self.kernel[:, self.hidden_dim * 4: self.hidden_dim * 5]   
        self.kernel6 = self.kernel[:, self.hidden_dim * 5:]   

    
    
    
##########################################################################
        self.kernel_all = self.add_weight(shape=(6*self.hidden_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
##########################################################################    
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.hidden_dim*6,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias1 = self.bias[:self.hidden_dim]
            self.bias2 = self.bias[self.hidden_dim: self.hidden_dim * 2]
            self.bias3 = self.bias[self.hidden_dim * 2: self.hidden_dim * 3]
            self.bias4 = self.bias[self.hidden_dim * 3: self.hidden_dim * 4]
            self.bias5 = self.bias[self.hidden_dim * 4: self.hidden_dim * 5]
            self.bias6 = self.bias[self.hidden_dim * 5:]
            
            
###########################################################################
            self.bias_all = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
    
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output1 = K.dot(inputs, self.kernel1)
        output2 = K.dot(inputs, self.kernel2) 
        output3 = K.dot(inputs, self.kernel3)
        output4 = K.dot(inputs, self.kernel4)
        output5 = K.dot(inputs, self.kernel5) 
        output6 = K.dot(inputs, self.kernel6)
 
        if self.use_bias:
            output1 = K.bias_add(output1, self.bias1, data_format='channels_last')
            output2 = K.bias_add(output2, self.bias2, data_format='channels_last')
            output3 = K.bias_add(output3, self.bias3, data_format='channels_last')
            output4 = K.bias_add(output4, self.bias4, data_format='channels_last')
            output5 = K.bias_add(output5, self.bias5, data_format='channels_last')
            output6 = K.bias_add(output6, self.bias6, data_format='channels_last')
            
        self.activation= activations.get('X_1')
        output1 = self.activation(output1)

        self.activation= activations.get('X_2')
        output2 = self.activation(output2)

        self.activation= activations.get('X_3')            
        output3 = self.activation(output3) 

        self.activation= activations.get('X_4')
        output4 = self.activation(output4)

        self.activation= activations.get('X_5')
        output5 = self.activation(output5)

        self.activation= activations.get('X_6')            
        output6 = self.activation(output6) 
        output_all=concatenate([output1,output2,output3,output4,output5,output6])

        output_all = K.dot(output_all, self.kernel_all)  
        output_all = K.bias_add(output_all, self.bias_all, data_format='channels_last')
        self.activation= activations.get('linear')
        output_all = self.activation(output_all)
            
        return output_all

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




#Define normalize Functions
def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

#Define Rescaling Functions from 0-1 to 0.1-0.9
def rescale_range(d):
    # d is a (n x dimension) np array
    d=np.multiply(d, 0.89)
    d=np.add(d, 0.01)
    return d
def build_model_v1(input_dim,hidden_dim,output_dim):
    model = Sequential()
    model.add(Dense_Co(output_dim,hidden_dim=hidden_dim ,input_dim=input_dim))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def build_model(input_dim,hidden_dim,output_dim):
    model = Sequential()
    model.add(Dense_Co(output_dim,hidden_dim=hidden_dim ,input_dim=input_dim,  kernel_initializer=RandomNormal(
            mean=0.0, stddev=0.04), bias_initializer=RandomNormal(mean=0.0, stddev=0.04)))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model
epochs=5
batch_size=1

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []
dataframe = read_csv('../input/uci-ionosphere/ionosphere_data_kaggle.csv', header=1)
dataset = dataframe.values

#Number of Rows and columns
Number_rows,Input_size=dataset.shape

X = dataset[:,0:Input_size-1].astype(float)
Y = dataset[:,Input_size-1]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

cvscores = []

# create model
model_1=build_model(Input_size-1,Input_size-1,1)
model_1.summary()

for train, test in kfold.split(X, Y):
    model_1.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_1.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_1.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
dataframe = read_csv('../input/iris/Iris.csv', header=1)
dataset = dataframe.values


#Number of Rows and columns
Number_rows,Input_size=dataset.shape

X = dataset[:,0:Input_size-1].astype(float)
Y = dataset[:,Input_size-1]
X=normalize(X)
X=rescale_range(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

cvscores = []

# create model
model_2=build_model(Input_size-1,Input_size-1,1)
model_2.summary()

for train, test in kfold.split(X, Y):
    model_2.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_2.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_2.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
dataframe = read_csv('../input/data_banknote_authentication/data_banknote_authentication.csv' , header=None)
dataset = dataframe.values


#Number of Rows and columns
Number_rows,Input_size=dataset.shape

X = dataset[:,0:Input_size-1].astype(float)
Y = dataset[:,Input_size-1]
X=normalize(X)
X=rescale_range(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

cvscores = []

# create model
model_3=build_model(Input_size-1,Input_size-1,1)
model_3.summary()

for train, test in kfold.split(X, Y):
    model_3.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_3.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_3.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
dataframe = read_csv('../input/mines-vs-rocks/sonar.all-data.csv', header=None)
dataset = dataframe.values


#Number of Rows and columns
Number_rows,Input_size=dataset.shape

X = dataset[:,0:Input_size-1].astype(float)
Y = dataset[:,Input_size-1]
X=normalize(X)
X=rescale_range(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

cvscores = []

# create model
model_4=build_model(Input_size-1,Input_size-1,1)
model_4.summary()

for train, test in kfold.split(X, Y):
    model_4.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_4.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_4.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
dataframe = read_csv('../input/pima-indians-diabetes-database/diabetes.csv', header=1)
dataset = dataframe.values


#Number of Rows and columns
Number_rows,Input_size=dataset.shape

X = dataset[:,0:Input_size-1].astype(float)
Y = dataset[:,Input_size-1]
X=normalize(X)
X=rescale_range(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

cvscores = []

# create model
model_5=build_model(Input_size-1,Input_size-1,1)
model_5.summary()

for train, test in kfold.split(X, Y):
    model_5.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_5.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_5.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
dataframe = read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv', header=1)
dataset = dataframe.values

#Number of Rows and columns
Number_rows,Input_size=dataset.shape

X = dataset[:,0:Input_size-1].astype(float)
Y = dataset[:,Input_size-1]
X=normalize(X)
X=rescale_range(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

cvscores = []

# create model
model_6=build_model(Input_size-1,Input_size-1,1)
model_6.summary()

for train, test in kfold.split(X, Y):
    model_6.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_6.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_6.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
df_XY_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_X_test  = pd.read_csv('../input/digit-recognizer/test.csv')

Y_train = df_XY_train['label'].values
X_train = df_XY_train.drop('label', axis=1).values
X_test  = df_X_test.values

# Normalize the data

# Normalize the data
X_train = X_train +1
test = test +1
X_train = X_train / 3.0
test = test / 3.0

img_rows, img_cols = 28, 28
input_shape = (784, 1) #tensorflow channels_last
num_classes = 10

X_train = X_train.reshape(X_train.shape[0],img_rows*img_cols).astype('float32')/255
Y_train = to_categorical(Y_train, num_classes)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=7)
# create model
epochs=10
batch_size=128
model_7=build_model(784,784,10)
model_7.summary()
model_7.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,validation_data = (X_val, Y_val))
from keras.models import Model,Sequential
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import SGD
from keras.layers import Input, Dense,concatenate,   Activation
epochs=5
batch_size=1
dataframe = read_csv('../input/pima-indians-diabetes-database/diabetes.csv', header=1)
dataset = dataframe.values


#Number of Rows and columns
Number_rows,Input_size=dataset.shape

X = dataset[:,0:Input_size-1].astype(float)
Y = dataset[:,Input_size-1]
X=normalize(X)
X=rescale_range(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
hidden_dimention=Input_size-1
in_ = Input(shape=(Input_size-1,))
intial=RandomNormal(mean = 0, stddev = 0.05)
model_8 = Sequential()

Layer_1_Act_X_1=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_1')(in_)
################################################################
Layer_1_Act_X_2=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_2')(in_)
###############################################################
Layer_1_Act_X_3=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_3')(in_)
###############################################################
Layer_1_Act_X_4=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_4')(in_)
###############################################################
Layer_1_Act_X_5=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_5')(in_)
###############################################################
Layer_1_Act_X_6=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_6')(in_)
###############################################################
Layer_1_Act_X_7=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_7')(in_)
###############################################################
Layer_1_Act_X_8=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_8')(in_)
###############################################################
Concatenate_First_Layer = concatenate([Layer_1_Act_X_1,Layer_1_Act_X_2,Layer_1_Act_X_3,Layer_1_Act_X_4,Layer_1_Act_X_5,Layer_1_Act_X_6,Layer_1_Act_X_7,Layer_1_Act_X_8])

Out_put_first_layer=Dense(hidden_dimention,kernel_initializer=intial, activation='linear')(Concatenate_First_Layer)

Out_put=Dense(1, activation='linear')(Out_put_first_layer)

model_8 = Model(in_ , Out_put)

model_8.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

Image("../input/image-/2layers.png")
cvscores = []

# create model
model_8=build_model(Input_size-1,Input_size-1,1)
model_8.summary()

for train, test in kfold.split(X, Y):
    model_8.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_8.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_8.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
model_9 = Sequential()

Layer_1_Act_X_1=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_1')(in_)
################################################################
Layer_1_Act_X_2=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_2')(in_)
###############################################################
Layer_1_Act_X_3=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_3')(in_)
###############################################################
Layer_1_Act_X_4=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_4')(in_)
###############################################################
Layer_1_Act_X_5=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_5')(in_)
###############################################################
Layer_1_Act_X_6=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_6')(in_)
###############################################################
Layer_1_Act_X_7=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_7')(in_)
###############################################################
Layer_1_Act_X_8=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial,  activation='X_8')(in_)

Concatenate_First_Layer = concatenate([Layer_1_Act_X_1,Layer_1_Act_X_2,Layer_1_Act_X_3,Layer_1_Act_X_4,Layer_1_Act_X_5,Layer_1_Act_X_6,Layer_1_Act_X_7,Layer_1_Act_X_8])

Out_put_first_layer=Dense(hidden_dimention,kernel_initializer=intial, activation='linear')(Concatenate_First_Layer)

###############################################################
Layer_2_Act_X_1=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_1')(Out_put_first_layer)
################################################################
Layer_2_Act_X_2=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_2')(Out_put_first_layer)
###############################################################
Layer_2_Act_X_3=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_3')(Out_put_first_layer)
###############################################################
Layer_2_Act_X_4=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_4')(Out_put_first_layer)
###############################################################
Layer_2_Act_X_5=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_5')(Out_put_first_layer)
###############################################################
Layer_2_Act_X_6=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_6')(Out_put_first_layer)
###############################################################
Layer_2_Act_X_7=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_7')(Out_put_first_layer)
###############################################################
Layer_2_Act_X_8=Dense(hidden_dimention,kernel_initializer=intial,bias_initializer=intial, activation='X_8')(Out_put_first_layer)
###############################################################
Concatenate_second_Layer = concatenate([Layer_2_Act_X_1,Layer_2_Act_X_2,Layer_2_Act_X_3,Layer_2_Act_X_4,Layer_2_Act_X_5,Layer_2_Act_X_6,Layer_2_Act_X_7,Layer_2_Act_X_8])
Out_put=Dense(1, activation='linear')(Concatenate_second_Layer)


model_9 = Model(in_ , Out_put)
model_9.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

Image("../input/image-/4layers.png")
cvscores = []

# create model
model_9=build_model(Input_size-1,Input_size-1,1)
model_9.summary()

for train, test in kfold.split(X, Y):
    model_9.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model_9.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model_9.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("The Mean accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# Python Program to Print 
# all subsets of given size of a set 

import itertools 

def findsubsets(s, n): 
	return list(itertools.combinations(s, n)) 

# Driver Code 
s = {1, 2, 3} 
n = 2

print(findsubsets(s, n)) 


# Python Program to Print 
# all subsets of given size of a set 

import itertools 

def findsubsets(s, n): 
	return list(itertools.combinations(s, n)) 

# Driver Code 
s = {'relu','sigmoid','tanh','selu'} 
n = 2

print(findsubsets(s, n)) 

temp=findsubsets(s, n)
# A Python program to print all 
# permutations using library function 
from itertools import permutations 

# Get all permutations of [1, 2, 3] 
perm = permutations([1, 2, 3]) 

# Print the obtained permutations 
for i in list(perm): 
	print (i) 

for j in temp:
    perm = permutations(j) 

    # Print the obtained permutations 
    for i in list(perm): 
        print (i)    
