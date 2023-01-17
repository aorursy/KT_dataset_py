# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import math

import tensorflow as tf

print(tf.__version__)



np.random.seed(2019)



import matplotlib.pyplot as plt

%matplotlib inline
data_train_file = "../input/Kannada-MNIST/train.csv"

data_test_file = "../input/Kannada-MNIST/test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)
df_train.head()
# Notice data is shifted to the left by one column, since the label is missing

df_test.head()
# Note this returns numpy arrays

def get_features_labels(df):

    # The first column is the label.

    labels = df['label'].values

    

    # Select all columns except the first

    features = df.values[:, 1:]/255

    

    return features, labels
def get_features_ids(df):

    # The first column is the id.

    labels = df['id'].values

    

    # Select all columns except the first

    features = df.values[:, 1:]/255

    

    return features, labels
train_features, train_labels = get_features_labels(df_train)

test_features, test_labels = get_features_ids(df_test)

# test_features = df_test.values/255
print(train_features.shape)

print(test_features.shape)

print(train_labels.shape)
# Defaults to showing data from the training set, 

# but we can provide the test data as well, and leave labels as None, to visualize test set

def display_by_index(index, features=train_features, labels=train_labels):

    plt.figure()

    

    if labels is not None:

        plt.title(f'Label: {labels[index]}')

        

    _ = plt.imshow(np.reshape(features[index, :], (28,28)), 'gray')
# Visualize a training sample

display_by_index(221)
# Visualize a test sample

display_by_index(221, features=test_features, labels=None)
df_train['label'].value_counts()
print(df_test.shape)
train_labels_1hot = tf.keras.utils.to_categorical(train_labels)
print(train_labels_1hot.shape)
model_arch = {}
model_arch['single_layer'] = [

      tf.keras.layers.Input(shape=(28*28,)),

      tf.keras.layers.Dense(10, activation='softmax')

  ]
model_arch['dnn'] = [

      tf.keras.layers.Input(shape=(28*28,)),

      tf.keras.layers.Dense(200, activation='sigmoid'),

      tf.keras.layers.Dense(60, activation='sigmoid'),

      tf.keras.layers.Dense(10, activation='softmax')

  ]
model_arch['dnn_relu'] = [

      tf.keras.layers.Input(shape=(28*28,)),

      tf.keras.layers.Dense(200, activation='relu'),

      tf.keras.layers.Dense(60, activation='relu'),

      tf.keras.layers.Dense(10, activation='softmax')

  ]
# lr decay function

def lr_decay(epoch):

    return 0.01 * math.pow(0.6, epoch)



# lr schedule callback

lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)



# Plot the decay rate

x = []

y = []

for i in range(1,10):

    y.append(lr_decay(i))

    x.append(i)

plt.plot(y, x)
# Add dropout

model_arch['dnn_relu_dropout'] = [

      tf.keras.layers.Input(shape=(28*28,)),

      tf.keras.layers.Dense(200, activation='relu'),

      tf.keras.layers.Dropout(0.25),

      tf.keras.layers.Dense(60, activation='relu'),

      tf.keras.layers.Dropout(0.25),

      tf.keras.layers.Dense(10, activation='softmax')

  ]
# CNN

model_arch['cnn'] = [

      tf.keras.layers.Reshape(input_shape=(28*28,), target_shape=(28, 28, 1)),

      tf.keras.layers.Conv2D(kernel_size=3, filters=12, activation='relu', padding='same'),

      tf.keras.layers.Conv2D(kernel_size=6, filters=24, activation='relu', padding='same', strides=2),

      tf.keras.layers.Conv2D(kernel_size=6, filters=32, activation='relu', padding='same', strides=2),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(200, activation='relu'),

      tf.keras.layers.Dense(10, activation='softmax')

  ]
model_arch.keys()
model = tf.keras.Sequential(model_arch['dnn_relu_dropout'])



optimizer = 'sgd'

# optimizer = 'adam'

# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)



# We will now compile and print out a summary of our model

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



model.summary()
BATCH_SIZE=128

EPOCHS=25
history = model.fit(train_features, train_labels_1hot, 

          epochs=EPOCHS, 

          batch_size=BATCH_SIZE, 

          validation_split=0.2,

          callbacks=[lr_decay_callback])
plt.plot(history.history['acc'], color='b', label="Training accuracy")

plt.plot(history.history['val_acc'], color='r', label="Validation accuracy")

plt.legend(loc='lower right', shadow=True)
predictions = model.predict_classes(test_features)



submissions=pd.DataFrame({"id": list(range(0,len(predictions))),

                         "label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)
for i in range(200,210):

    display_by_index(i, features=test_features, labels=submissions["Label"].values)
