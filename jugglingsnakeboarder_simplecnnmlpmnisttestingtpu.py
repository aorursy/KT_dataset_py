# Detect hardware, return appropriate distribution strategy

import tensorflow as tf

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
pwd
import numpy as np # numeric python

import pandas as pd # data management

import seaborn as sbn # data visualisation

import matplotlib.pyplot as pplt

%matplotlib inline

from keras.models import Sequential

from keras.layers.core import Dense

from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.regularizers import l2

from keras.utils import np_utils
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv") 

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
test.head()
# seperate the first column as Output-Data

YTrain = train["label"] 

# cut the 'label' column from training-data

XTrain = train.drop(labels = ["label"],axis = 1)  

del train

g = sbn.countplot(YTrain) 

YTrain.value_counts()
# normalisation to [0..1] instead of [0..255]

XTrain = XTrain / 255.0 

test = test / 255.0
# keras needs a shape (height,weight,channel)

# whith channel=1=greyscale, =3=rgb

XTrain = XTrain.values.reshape(-1,28,28,1) # -1 means all samples

print(XTrain.shape)

test = test.values.reshape(-1,28,28,1)

# the lables as "one-hot"-coding 

YTrain = np_utils.to_categorical(YTrain, num_classes = 10)

print(YTrain.shape)
# like to see the first digit

myfirstdigit = XTrain[0]

pplt.imshow(myfirstdigit [:,:,0],cmap=pplt.cm.binary)

pplt.show()
# showing the last digit

mylastdigit = XTrain[41999]

pplt.imshow(mylastdigit [:,:,0],cmap=pplt.cm.binary)

pplt.show()
# create the train- and validation-data

from sklearn.model_selection import train_test_split

XTrain, XVal, YTrain, YVal = train_test_split(XTrain, YTrain, test_size=0.1, random_state=None )

print("XTrain shape: ",XTrain.shape)

print("YTrain shape: ",YTrain.shape)

print("XVal shape: ",XVal.shape)

print("YVal shape: ",YVal.shape)
# instantiating the model in the strategy scope creates the model on the TPU



with strategy.scope():

    model = tf.keras.Sequential([

         tf.keras.layers.Conv2D(32,(4,4),padding='same',activation='relu',kernel_regularizer=l2(0.001),activity_regularizer=l2(0.001),input_shape=(28,28,1)), 

         tf.keras.layers.MaxPool2D((2,2)),

         tf.keras.layers.Conv2D(64,(4,4),padding='same',activation='relu',kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01)),

         tf.keras.layers.MaxPool2D((2,2)),

         tf.keras.layers.Conv2D(128,(4,4),padding='same',activation='relu',kernel_regularizer=l2(0.001),activity_regularizer=l2(0.001)),

         tf.keras.layers.MaxPool2D((2,2)),

         tf.keras.layers.Flatten(),

         tf.keras.layers.Dense(128,activation='relu'),#128

         tf.keras.layers.Dense(10,activation='softmax'),

     ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    model.summary()
#history=CNN.fit(XTrain,YTrain,epochs=6)

my_batch_size = 8  * strategy.num_replicas_in_sync *16 

my_steps = len(XTrain) // my_batch_size

history = model.fit(XTrain.astype(np.float32),YTrain.astype(np.float32),

                    shuffle=True,

                    epochs=200,

                    batch_size=my_batch_size,

                    steps_per_epoch=my_steps,

                    validation_data=(XVal.astype(np.float32),YVal.astype(np.float32))# astype float32 seems to be very important for TPU processing

                    )
history_dict = history.history

history_dict.keys()# look witch keys exist

loss_values = history_dict['loss']

acc_values = history_dict['accuracy']

epochs = range(1, len(loss_values)+1)

pplt.plot(epochs,loss_values,'r', label='training loss')

pplt.plot(epochs,acc_values,'g',label='accuracy')

pplt.title('loss and accuracy')

pplt.xlabel('epochs')

pplt.ylabel('loss vs. accuracy')

pplt.legend()

pplt.show()
# prediction

results = model.predict(test.astype(np.float32)) 

# index with the maximum probability

results = np.argmax(results,axis = 1) 

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1) 

submission.to_csv("mysubmission.csv",index=False) 