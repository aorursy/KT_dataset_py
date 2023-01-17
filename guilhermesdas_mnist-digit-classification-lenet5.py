# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

from tensorflow.keras.regularizers import l2
class LeNet5(Model):

  '''

  LeNet5 Model



  Attributes

  -----------------

  conv1: tf.keras.layers

    Convolutional layer of model

  max_pool1: tf.keras.layers

    Maxpooling layer of model

  conv2: tf.keras.layers

    Convolutional layer of model

  max_pool2: tf.keras.layers

    Maxpooling layer of model

  flatten: tf.keras.layers

    Flatten layer of model 

  dense1: tf.keras.layers

    Dense layer of model

  dense2: tf.keras.layers

    Dense layer of model

  dense3: tf.keras.layers

    Output Layer of model

  '''



  def __init__(self, num_classes, reg=0):

    '''

    Initialize the model

    :param num_classes: number of classes to predict from

    '''



    super(LeNet5, self).__init__()

    #building the various layers that compose our LeNet-5:

    self.conv1 = Conv2D(6, kernel_size=(5,5), padding='same', activation='relu', activity_regularizer=l2(reg))

    self.max_pool1 = MaxPooling2D(pool_size=(2,2))

    self.conv2 = Conv2D(16, kernel_size=(5,5), activation='relu', activity_regularizer=l2(reg))

    self.max_pool2 = MaxPooling2D(pool_size=(2,2))

    self.flatten = Flatten()

    self.dense1 = Dense(120, activation='relu', activity_regularizer=l2(reg))

    #self.dropout = Dropout(0.2)

    self.dense2 = Dense(84, activation='relu', activity_regularizer=l2(reg))

    self.dense3 = Dense(num_classes, activation='softmax', activity_regularizer=l2(reg))



  def call(self, inputs):

    '''

    Call the layers and perform their operations on the input tensors

    :param inputs:  Input tensor

    :return:        Output tensor

    '''

    x = self.max_pool1(self.conv1(inputs)) # 1st block

    x = self.max_pool2(self.conv2(x)) # 2nd block

    x = self.flatten(x) 

    x = self.dense2(self.dense1(x))

    x = self.dense3(x) #fully conected layers

    return x
num_classes = 10

img_rows, img_cols, img_ch = 28, 28, 1

input_shape = (img_rows,img_cols,img_ch)



# load data

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

X_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
Y_train = train.iloc[:,0]

X_train = train.iloc[:,1:]



X_train = X_train.to_numpy()

X_test = X_test.to_numpy()

Y_train = Y_train.to_numpy()



temp = X_train.reshape(-1,28,28)

X_test = X_test.reshape(-1,28,28)



X_train = temp.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)



X_train = X_train/255.

X_test = X_test/255.



Y_train = tf.keras.utils.to_categorical(Y_train)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)
# compile model

model = LeNet5(num_classes, 1e-5)

model.compile(  tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy',

                metrics = 'accuracy' )



# create callbacks

callbacks = [ 

  tf.keras.callbacks.ModelCheckpoint('best_model', monitor='val_accuracy', verbose=1, save_best_only=True),

  tf.keras.callbacks.ReduceLROnPlateau( factor = 0.1, patience = 3, min_lr = 0.00001, verbose = 1 )

 ]



history = model.fit(X_train, Y_train, epochs = 50, batch_size = 256,

                    callbacks = callbacks, verbose = 1,

                    validation_split=0.2 )
scores = model.evaluate(X_train, Y_train)
y_pred = model.predict(X_test)

print(y_pred.shape)
preds = np.argmax(y_pred,axis=1)

preds.shape
y_pred.shape[0]
y_pred.shape[0]

output = pd.DataFrame({'ImageId': [x for x in range(1,y_pred.shape[0]+1)], 'Label': preds})

output.head()
output.to_csv('Y_test.csv', index=False)

model.save("saved_model")