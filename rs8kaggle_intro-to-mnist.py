import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
print("Number of training examples :", len(train))
print("Number of test examples :", len(test))
train_labels = train['label']
train_data = train.drop(['label'], axis=1)
test_data = test

train_data.shape, test_data.shape
trainDistribution = pd.DataFrame(train_labels.value_counts(sort=False))
trainDistribution.columns = ['MNIST Train Examples Count']
trainDistribution
plt.figure()
sample_image = np.array(train_data.iloc[3]).reshape([28,28])
plt.imshow(sample_image)
plt.grid(True)
train_data = np.array(train_data) / 255.0
train_labels = np.array(train_labels)
test_data  = np.array(test_data) / 255.0
plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(train_data[i].reshape([28,28]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
nn_model1 = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
nn_model1.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
nn_model1.fit(train_data, train_labels, epochs=3)
nn_model2 = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
     keras.layers.Dense(64, activation=tf.nn.relu),
     keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
nn_model2.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
nn_model2.fit(train_data, train_labels, epochs=5)
predictions_on_model1 = nn_model1.predict(test_data)
predictions_on_model1 = np.argmax(predictions_on_model1, axis=1)
predictions_on_model1
plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(test_data[i].reshape([28,28]), cmap=plt.cm.binary)
    plt.xlabel(predictions_on_model1[i])
predictions_on_model2 = nn_model2.predict(test_data)
predictions_on_model2 = np.argmax(predictions_on_model2, axis=1)
predictions_on_model2
plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(test_data[i].reshape([28,28]), cmap=plt.cm.binary)
    plt.xlabel(predictions_on_model2[i])
predictions = pd.Series(predictions_on_model2)
predictions = predictions.to_frame(name='Label')
predictions.set_index(np.arange(1, len(predictions) + 1), inplace=True)
predictions.to_csv('submission.csv',index_label='ImageId')
predictions