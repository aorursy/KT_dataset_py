# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#get dataset

import pandas as pd
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
print("train dataset size: ", train_dataset.shape[0])
print("test dataset size: ", test_dataset.shape[0])

#split train dataset
import numpy
msk = np.random.rand(len(train_dataset)) < 0.75
train = train_dataset[msk]
cv = train_dataset[~msk]
print("train dataset size: ", train.shape[0])
print("cross validation dataset size: ", cv.shape[0])
#define labels and normalize inputs
labels_train = train["label"]
X_train = np.array(train.drop("label",axis=1)) / 255
labels_cv = cv["label"]
X_cv = np.array(cv.drop("label",axis=1)) / 255

#convert labels to multi-class binaries
from sklearn import preprocessing 
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(labels_train)
y_cv = lb.fit_transform(labels_cv)

#explole the range of numbers in dataset
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
ax.pie(labels_train.value_counts(),labels = ['0','1','2','3','4','5','6','7','8','9'])
fig, ax1 = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
ax1.pie(labels_cv.value_counts(),labels = ['0','1','2','3','4','5','6','7','8','9'])
plt.show()

#define constant variables
num_of_train_examples = X_train.shape[0]
num_of_features = X_train.shape[1]
num_of_cv_examples = X_cv.shape[0]
num_of_classes = 10


#initialize parameters
w = np.zeros(num_of_features)
b = 0
parameters = {
    "weights" : w,
    "bias" : b
}

#initialize hyperparameters
num_iterations = 100
reg_lambda = 0.005
learning_rate = 0.08
hyperparameters = {
    "num_iterations" : num_iterations,
    "lambda" : reg_lambda,
    "alpha" : learning_rate
}
# Compute Hypothesis (Predict)
def forward_propagation(X,parameters):
    w = parameters.get("weights")
    b = parameters.get("bias")
    z = np.dot(X,w) + b
    h = 1 / (1 + np.exp(-z))
    return h

# compute Cost Function
def compute_log_reg_cost(h,y,m):
    return -(np.dot(y.T,np.log(h)) + np.dot((1-y.T),np.log(1-h))) / m

#compute gradients
def backward_propagation(X,y,h,hyperparameters,m):
    learning_rate = hyperparameters.get("alpha")
    
    error = h-y
    grad_w = (learning_rate / m) * np.dot(X.T,error)
    grad_b = (learning_rate / m) * sum(error)
    
    gradients = {
        "dw" : grad_w,
        "db" : grad_b
    }
    
    return gradients

#update parameters
def update_parameters(parameters,gradients):
    dw = gradients.get("dw")
    db = gradients.get("db")
    w = parameters.get("weights")
    b = parameters.get("bias")

    w = w - dw
    b = b - db
    
    parameters = {
        "weights" : w,
        "bias" : b
    }
    
    return parameters

#model
def logistic_regression_model(X,y,m,n,parameters,hyperparameters):
    num_iterations = hyperparameters.get("num_iterations")
    cost_list = []
    for i in range(num_iterations):
        hypothesis = forward_propagation(X,parameters)
        cost = compute_log_reg_cost(hypothesis,y,m)
        gradients = backward_propagation(X,y,hypothesis,hyperparameters,m)
        parameters = update_parameters(parameters,gradients)
        if i%20 == 0:
            cost_list.append(cost)
    
    return (parameters,cost_list)

#run model for each label
def multi_label_logistic_regression_model(X,Y,num_labels,m,n,parameters,hyperparameters):
    W = []
    B = []
    for i in range(num_labels):
        y = Y[: , i]
        parameters,cost = logistic_regression_model(X,y,m,n,parameters,hyperparameters)
        W.append(parameters.get("weights"))
        B.append(parameters.get("bias"))
    multi_parameters = {
        "W" : np.array(W),
        "B" : np.array(B)
    }
    return multi_parameters

multi_parameters = multi_label_logistic_regression_model(X_train,y_train,num_of_classes,num_of_train_examples,num_of_features,parameters,hyperparameters)
#predict - return the label with the bigger propability
def predict(X,multi_parameters):
    W = multi_parameters.get("W")
    B = multi_parameters.get("B")
    z = np.dot(X,W.T) + B
    predict = 1 / (1 + np.exp(-z))
    max_idx = predict.argmax(axis=1)
    return max_idx
train_predictions = predict(X_train,multi_parameters)
cv_predictions = predict(X_cv,multi_parameters)
#evaluate results
def evaluation(predictions,Y):
    total = Y.shape[0]
    count_right_predict = 0
    idx = 0
    for predict in predictions:
        if predict == Y[idx] : 
            count_right_predict = count_right_predict + 1
        idx = idx + 1
    return (count_right_predict/total)
train_accuracy = evaluation(train_predictions,np.array(labels_train))
cv_accuracy = evaluation(cv_predictions,np.array(labels_cv))
print('train_accuracy: ', train_accuracy)
print('cv_accuracy: ' , cv_accuracy)
batch_size = 128

n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 128 # 3nd layer number of neurons

import tensorflow as tf
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'digits_images': np.array(X_train)}, y=np.array(labels_train),
    batch_size=batch_size,  shuffle=True)

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['digits_images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    layer_3 = tf.layers.dense(layer_2, n_hidden_3)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_3, num_of_classes)
    return out_layer
# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    learning_rate = 0.1
    
    # Build the neural network
    logits = neural_net(features)
    
    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)
    
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
        
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    
    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)


# Train the Model
model.train(input_fn, steps=num_iterations)

#evaluate
model.evaluate(input_fn)


# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'digits_images': np.array(X_cv)}, y=np.array(labels_cv),
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
model.evaluate(input_fn)


X_train = X_train.reshape(-1, 28,28, 1)
X_cv = X_cv.reshape(-1, 28,28, 1)
X_train.shape, X_cv.shape
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_of_classes, activation='softmax'))
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
fashion_model.summary()
num_iterations = 5
fashion_train = fashion_model.fit(X_train, y_train, batch_size=batch_size,epochs=num_iterations,verbose=1,validation_data=(X_cv, y_cv))
