from tensorflow import keras
keras.__version__
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output
df_train_file = "../input/fashion-mnist_train.csv"
df_test_file = "../input/fashion-mnist_test.csv"
df_train = pd.read_csv(df_train_file)
df_test = pd.read_csv(df_test_file)

df_train.head()
def get_features_label(df):
    features = df.values[:,1:]/255
    
    labels = df['label'].values
    return features, labels
train_features, train_labels = get_features_label(df_train)
test_features, test_labels = get_features_label(df_test)
print(train_features.shape)
print(test_features.shape)
#take a peak at some values in an image
train_features[20, 300:320]
example_index = 221
plt.figure()
_ = plt.imshow(np.reshape(train_features[example_index, :],(28,28)), 'gray')
train_labels.shape
train_labels[example_index]
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
train_labels.shape
train_labels[example_index]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation = tf.nn.relu, input_shape = (784,)))
model.add(tf.keras.layers.Dense(20, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

# compile and print summary of model

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'rmsprop',
              metrics = ['accuracy'])
model.summary()
EPOCHS = 2
BATCH_SIZE = 128
model.fit(train_features, train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE)
test_loss, test_acc = model.evaluate(test_features, test_labels)
print("test accuracy: ", test_acc)
