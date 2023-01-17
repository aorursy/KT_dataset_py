#set up kaggle.com API

from google.colab import files

files.upload()



!pip install -q kaggle==1.5.4



!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!ls ~/.kaggle

!chmod 600 /root/.kaggle/kaggle.json  # set permission



!kaggle competitions download -c digit-recognizer
!unzip test.csv.zip

!unzip train.csv.zip
!head -5 train.csv
!cat train.csv | wc -l
%tensorflow_version 2.x

import pandas as pd

import numpy as np



!pip install -q keras==2.3.0

import keras

from keras import regularizers

from keras import optimizers

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.regularizers import l1, l2

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



!pip install -q livelossplot

from livelossplot import PlotLossesKeras



from time import time

import matplotlib.pyplot as plt

%matplotlib inline
import tensorflow as tf

tf.test.gpu_device_name()
df_train = pd.read_csv('train.csv', sep =',')

display(df_train.columns)

df_train.head()
for i in df_train.columns:

  if df_train[i].isna().any() == True:

    print('in {} is a null'.format(i))

else: print('all ok')
df_train['label'].hist()
#split data

train_x = df_train.sample(frac=0.79, random_state=1)

test_x = df_train.drop(train_x.index)



#pop targets

train_y = to_categorical(train_x.pop('label'))

test_y = to_categorical(test_x.pop('label'))



train_x = train_x/255

test_x = test_x/255



train_x = train_x.values.reshape(-1,28,28,1) #-1 means we want numpy to figure out one dimension one channel for grayscale

test_x = test_x.values.reshape(-1,28,28,1)



plt.imshow(train_x[1][:,:,0])
train_x.shape, test_x.shape, train_y.shape, test_y.shape
def model_v3():

  

  model = keras.models.Sequential()

 

  model.add(keras.layers.Conv2D(filters = 64, kernel_size = (5,5), strides = (1, 1), padding = 'same', input_shape = train_x.shape[1:4], 

                                data_format= 'channels_last', dilation_rate = (1,1), activation='relu', use_bias=True, 

                                kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', kernel_regularizer= None, 

                                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)) 

  model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

  model.add(Dropout(0.2))



  model.add(keras.layers.Conv2D(filters = 64, kernel_size = (4,4), strides = (1,1), padding = 'same', input_shape = train_x.shape[1:4], 

                                data_format= 'channels_last', dilation_rate = (1,1), activation='relu', use_bias=True, 

                                kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', kernel_regularizer= None, 

                                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)) 

  model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

  model.add(Dropout(0.2))



  model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same', input_shape = train_x.shape[1:4], 

                                data_format= 'channels_last', dilation_rate = (1,1), activation='relu', use_bias=True, 

                                kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', kernel_regularizer= None, 

                                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)) 

  model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

  model.add(Dropout(0.2))

  

  #FLATTEN

  model.add(Flatten())



  model.add(Dense(1000, kernel_initializer = 'TruncatedNormal', use_bias= True, bias_initializer='TruncatedNormal', activation='relu', 

                  activity_regularizer = regularizers.l1(5*10**(-5))))

  model.add(Dropout(0.5))



  model.add(Dense(500, kernel_initializer = 'TruncatedNormal', use_bias= True, bias_initializer='TruncatedNormal', activation='relu', 

                  kernel_regularizer = None))



  #output-softmax

  model.add(Dense(train_y.shape[1], activation='softmax'))



 #compile

  model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



  return model



model_v3 = model_v3()



model_v3.summary()
start = time()

history_v3 =  model_v3.fit(train_x, train_y.astype('int'), 

                           epochs=50, shuffle = True, 

                           validation_split = 0.2, 

                           workers=1, 

                           callbacks=[PlotLossesKeras()], verbose=0)



score_v3 = model_v3.evaluate(x = test_x, y = test_y, batch_size=32, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=12, use_multiprocessing=False)

print('\n score: {} \n'.format(dict(zip(model_v3.metrics_names, score_v3))))

print('\n run time:', (time()-start)//60, 'minutes' )
x_test = pd.read_csv('test.csv')

x_test = x_test/255

x_test = x_test.values.reshape(-1,28,28,1) #-1 means we want numpy to figure out one dimension

prediction_v3 = model_v3.predict(x_test)
prediction_v3
submission_v3 = np.argmax(prediction_v3, axis = -1)

submission_v3
len(submission_v3)
type(submission_v3)
submission_v3 = pd.DataFrame(submission_v3, columns = ['Label'])

submission_v3.head()
submission_v3['ImageId'] = np.arange(1,len(submission_v3)+1,1)



submission_v3.head()
submission_v3.to_csv("submission_v3_15.csv", index=False, header=True)
!kaggle competitions submit digit-recognizer -f submission_v3_15.csv -m "CNN"