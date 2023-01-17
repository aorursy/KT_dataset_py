from numpy.random import seed
seed(8)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
%matplotlib inline
import tensorflow as tf

from sklearn.datasets import load_iris
iris_data = load_iris()
print(iris_data['DESCR'])
### load and prepreocess the data
def read_in_and_split_data(iris_data):
    
    data = iris_data['data']
    target = iris_data['target']
    
    train_data, test_data, train_targets, test_targets = train_test_split(data, target, test_size = 0.1)
    return train_data, test_data, train_targets, test_targets
# Generate the test and training data.
iris_data = datasets.load_iris()
train_data, test_data, train_targets, test_targets = read_in_and_split_data(iris_data)
# Convert targets to a one-hot encoding
train_targets = tf.keras.utils.to_categorical(np.array(train_targets))
test_targets = tf.keras.utils.to_categorical(np.array(test_targets))
# the model 
def get_model(input_shape):
    """
    This function should build a Sequential model according to the above specification. Ensure the 
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    Your function should return the model.
    """
    model = Sequential([
        Dense(64, activation = 'relu', 
              #initializer = tf.keras.initializers.HeUniform(), 
              bias_initializer=tf.keras.initializers.Ones(),
              input_shape = input_shape),
        Dense(128, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(3,  activation = 'softmax')    
    ])
    return model
# get the model
model = get_model(train_data[0].shape)
# compile the model
def compile_model(model):
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])
# fit the model to the training data
def train_model(model, train_data, train_targets, epochs):
    history = model.fit(train_data, train_targets, epochs = epochs, validation_split = 0.15, 
                        batch_size = 40 )
    return history
# compile the model
compile_model(model)
# Run to train the model
history = train_model(model, train_data, train_targets, epochs=800)
#plot the learning curves 
try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
except KeyError:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 

# regularized model 
def get_regularised_model(input_shape, dropout_rate, weight_decay):
    
    model = Sequential([
        Dense(128, activation="relu", kernel_regularizer = regularizers.l2(weight_decay), input_shape=input_shape),
        Dense(128, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(128, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(128, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(128, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        BatchNormalization(),
        Dense(64, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(64, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(64, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(64, activation="relu", kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(3, activation="softmax")
    ])
    return model
# Let's now instantiate the model, using a dropout rate of 0.3 and weight decay coefficient of 0.001
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

#istantiate
reg_model = get_regularised_model(train_data[0].shape, 0.3, 0.001)

#compile
compile_model(reg_model)

# train 
reg_history = train_model(reg_model, train_data, train_targets, epochs=800)
try:
    plt.plot(reg_history.history['accuracy'])
    plt.plot(reg_history.history['val_accuracy'])
except KeyError:
    plt.plot(reg_history.history['acc'])
    plt.plot(reg_history.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show() 
plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 

