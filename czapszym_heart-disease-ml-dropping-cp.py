# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from __future__ import absolute_import, division, print_function



import pandas as pd



import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import contrib

tfe = contrib.eager

print(tf.__version__)

tf.enable_eager_execution()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
csv_path = "../input/heart.csv"

column_names = ['age', 'sex','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca', 'thal', 'target']

feature_names = column_names[:-1]

label_name = column_names[-1]

class_names = ['Healthy', 'Sick']

data = pd.read_csv(csv_path)

data.drop('cp', axis=1)

features=tf.convert_to_tensor(data[feature_names].astype('float32').values)

labels = tf.convert_to_tensor(data[label_name].astype('int32').values)

validation_features1 = features[:30]

validation_labels1 = labels[:30]

validation_features2 = features[-30:]

validation_labels2 = labels[-30:]

test_features1= features[30:60]

test_labels1 = labels[30:60]

test_features2 = features[-60:-30]

test_labels2 = labels[-60:-30]



features = features[60:-60]

labels = labels[60:-60]

model = tf.keras.Sequential([

  tf.keras.layers.Dense(15, activation=tf.nn.relu, input_shape=(12,)),  # input shape required

  tf.keras.layers.Dense(12, activation=tf.nn.softmax),

    tf.keras.layers.Dense(7, activation=tf.nn.relu),

  tf.keras.layers.Dense(2)

])

def loss(model, x, y):

    y_ = model(x)

    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
def grad(model, inputs, targets):

    with tf.GradientTape() as tape:

        loss_value = loss(model, inputs, targets)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)



global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)



print("Step: {}, Initial Loss: {}".format(global_step.numpy(),

                                          loss_value.numpy()))



optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)



print("Step: {},         Loss: {}".format(global_step.numpy(),

                                          loss(model, features, labels).numpy()))

train_loss_results = []

train_accuracy_results = []



num_epochs = 450



for epoch in range(num_epochs):

    epoch_loss_avg = tfe.metrics.Mean()

    epoch_accuracy = tfe.metrics.Accuracy()



    loss_value, grads = grad(model, features, labels)

    optimizer.apply_gradients(zip(grads, model.trainable_variables),

                              global_step)

    epoch_loss_avg(loss_value) 

    epoch_accuracy(tf.argmax(model(features), axis=1, output_type=tf.int32), labels)



    train_loss_results.append(epoch_loss_avg.result())

    train_accuracy_results.append(epoch_accuracy.result())

  

    if epoch % 20 == 0:

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,

                                                                epoch_loss_avg.result(),

                                                                epoch_accuracy.result()))

    

print("Below should be 1:\n")

validation_features1=np.array(validation_features1)

ynew = model.predict_classes(np.array(validation_features1))

# show the inputs and predicted outputs

wrong_zeros = 0

for i in range(len(validation_features1)):

    if(ynew[i] == 0):

        wrong_zeros +=1

    print("Predicted=%s" % ( ynew[i]))

print("Below should be 0:\n")

validation_features2=np.array(validation_features2)

ynew = model.predict_classes(np.array(validation_features2))

# show the inputs and predicted outputs

wrong_ones = 0

for i in range(len(validation_features2)):

    if(ynew[i] == 1):

        wrong_ones +=1

    print("Predicted=%s" % ( ynew[i]))



print("False positives: ", wrong_zeros)

print("False negatives: ", wrong_ones)

true_positives = len(validation_features1) - wrong_zeros

false_positives = wrong_ones

false_negatives = wrong_zeros

precision =true_positives/ (true_positives + false_positives)

recall = true_positives / (false_negatives + true_positives)

print("precision: ",precision)

print("recall: ",recall)

print("F1 score", 2*(precision*recall)/(precision + recall))



print("Below should be 1:\n")

validation_features1=np.array(test_features1)

ynew = model.predict_classes(np.array(test_features1))

# show the inputs and predicted outputs

wrong_zeros = 0

for i in range(len(test_features1)):

    if(ynew[i] == 0):

        wrong_zeros +=1

    print("Predicted=%s" % ( ynew[i]))

print("Below should be 0:\n")

validation_features2=np.array(test_features2)

ynew = model.predict_classes(np.array(test_features2))

# show the inputs and predicted outputs

wrong_ones = 0

for i in range(len(test_features2)):

    if(ynew[i] == 1):

        wrong_ones +=1

    print("Predicted=%s" % ( ynew[i]))



true_positives = len(test_features1) - wrong_zeros

false_positives = wrong_ones

false_negatives = wrong_zeros

precision =true_positives/ (true_positives + false_positives)

recall = true_positives / (false_negatives + true_positives)

print("precision: ",precision)

print("recall: ",recall)

print("F1 score", 2*(precision*recall)/(precision + recall))


