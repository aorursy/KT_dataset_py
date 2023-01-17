# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import tensorflow as tf

from tensorflow.keras.regularizers import l1

from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.



print(tf.__version__)
train = pd.read_csv('../input/learn-together/train.csv')

train.head()
test = pd.read_csv('../input/learn-together/test.csv')

test_Id = test['Id'] 
train_stats = train.describe()

train_stats = train_stats.transpose()

train_stats
train = train.drop(['Soil_Type7','Soil_Type15','Id'], axis =1)

test = test.drop(['Soil_Type7','Soil_Type15','Id'], axis =1)
train_labels = train.pop('Cover_Type') -1

train.head()
train_labels.head()
def maxNorm (df, themin,themax):

    normalized_df=(df-themin)/( themax - themin)

    return normalized_df
train_min = train.min()

train_max = train.max()

normed_train_data = maxNorm(train, train_min, train_max)

normed_test_data = maxNorm(test,train_min, train_max)

normed_train_data.head()
print (len(train_labels))

uniqueValues = np.unique(train_labels)

print('Unique Values : ',uniqueValues)

cnt_labels = len(uniqueValues)

print (cnt_labels)
def build_model(length):

    model = keras.Sequential([

    layers.Dense(512, activation=tf.nn.relu, input_shape=[length]), 

    layers.Dropout(0.5),

    layers.Dense(256, activation=tf.nn.relu, activity_regularizer=l2(0.001)),   

    layers.Dropout(0.5),

    layers.Dense(256, activation=tf.nn.relu, activity_regularizer=l2(0.001)), 

    layers.Dropout(0.5),

    layers.Dense(7, activation=tf.nn.softmax)

    ])

    #optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)

    optimizer = tf.keras.optimizers.Adam()

    

    model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy',

              metrics=['sparse_categorical_accuracy' ])

    return model



model = build_model(len(normed_train_data.keys()))

model.summary()
good = 0

bad=0

predictions = model.predict(normed_train_data)

for i in range(len(normed_train_data)):

    if (np.argmax(predictions[i]) == train_labels[i]):

        good +=1

    else:

        bad +=1

print ("good:", good)

print ("bad:", bad)
es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=40,restore_best_weights=False )

history = model.fit(normed_train_data, train_labels,batch_size=1024,callbacks=[es],

                    epochs=350, validation_split = 0.2, verbose=1)
_, train_acc = model.evaluate(normed_train_data, train_labels, verbose=0)

print('Train: %.3f' % (train_acc))

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

bestEpochs = hist.tail(1).epoch.item()
def plot_history(history, val=True):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Accuracy ')

  plt.plot(hist['epoch'], hist['sparse_categorical_accuracy'],

           label='Train Accuracy')

  if val:

     plt.plot(hist['epoch'], hist['val_sparse_categorical_accuracy'],label = 'Val Accuracy')

  plt.ylim([0,1])

  plt.legend()

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Loss ')

  plt.plot(hist['epoch'], hist['loss'],

           label='Train Loss')

  if val:

      plt.plot(hist['epoch'], hist['val_loss'],label = 'Val Loss')

  plt.ylim([0,1])

  plt.legend()





  plt.show()





plot_history(history)
finalmodel = build_model(len(normed_train_data.keys()))

history = finalmodel.fit(normed_train_data, train_labels,batch_size=1024,

                    epochs=bestEpochs + 1, verbose=1)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

example_batch = normed_train_data[:10]

example_result = finalmodel.predict(example_batch)

example_result
good = 0

bad=0

predictions = finalmodel.predict(normed_train_data)

for i in range(len(normed_train_data)):

    if (np.argmax(predictions[i]) == train_labels[i]):

        good +=1

    else:

        bad +=1

print ("good:", good)

print ("bad:", bad)
plot_history(history,False)
test_predictions = finalmodel.predict(normed_test_data)
print (test_predictions[0])

test_label = np.argmax(test_predictions,axis=1) + 1

print (test_label[0])
submission = pd.DataFrame(columns=['Id', 'Cover_Type'])

submission['Id'] = test_Id

submission['Cover_Type'] = test_label

submission.to_csv('submission.csv', index=False)