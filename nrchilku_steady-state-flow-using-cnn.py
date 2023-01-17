import os
import numpy as np
import tensorflow as tf
from glob import glob as glb
from tqdm import tqdm_notebook as tqdm

def read_and_decode(filename_queue, shape):
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'boundary':tf.FixedLenFeature([],tf.string),
      'sflow':tf.FixedLenFeature([],tf.string)
    }) 
  boundary = tf.decode_raw(features['boundary'], tf.uint8)
  sflow = tf.decode_raw(features['sflow'], tf.float32)
  boundary = tf.reshape(boundary, [1, shape[0], shape[1]])
  sflow = tf.reshape(sflow, [shape[0], shape[1], 2])
  boundary = tf.to_float(boundary)
  sflow = tf.to_float(sflow) 
  return boundary, sflow 

with tf.Session() as sess:
  filename_queue = tf.train.string_input_producer(["../input/train.tfrecords"])
  boundary, sflow = read_and_decode(filename_queue, (128, 256))

  init_op = tf.initialize_all_variables()
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
    
  for i in tqdm(range(2440)):
    example, l = sess.run([boundary, sflow])
    l = l[:, :, 0]**2 + l[:, :, 1]**2
    l = np.reshape(l, (1, 128, 256))
    if i == 0:
        train_x = example
        train_y = l
    else:
        train_x = np.concatenate((train_x, example), axis=0)
        train_y = np.concatenate((train_y, l), axis=0)
    #print (example.shape,l.shape)
  coord.request_stop()
  coord.join(threads)
train_x = np.reshape(train_x, (2440, 128, 256, 1))
train_y = np.reshape(train_y, (2440, 128, 256, 1))
print(train_x.shape, train_y.shape)
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Dense, MaxPooling2D, Flatten, Conv2DTranspose
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

# numpy and matplot lib
import numpy as np
import matplotlib.pyplot as plt

# training params
batch_size = 16
epochs = 50 # number of times through training set



# construct model
inputs = Input(train_x.shape[1:])
conv1 = Conv2D(128, (8, 16), strides=(8, 16), activation='relu')(inputs)
conv1 = Conv2D(512, (4, 4), strides=(4, 4), activation='relu', padding='valid')(conv1)
dense = Conv2D(1024, (4, 4), activation='relu', padding='valid')(conv1)
conv2 = Conv2DTranspose(512, (8, 8), strides=(8, 8), activation='relu', padding='valid')(dense)
conv2 = Conv2DTranspose(256, (4, 8), strides=(4, 8), activation='relu', padding='valid')(conv2)
conv5 = Conv2DTranspose(32, (2, 2),  strides=(2, 2), activation='relu', padding='valid')(conv2)
conv5 = Conv2DTranspose(1, (2, 2), strides=(2, 2), activation='linear', padding='valid')(conv5)

# construct model
model = Model(inputs=[inputs], outputs=[conv5])
model.summary()
# compile the model with loss and optimizer
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['MSE'])

# train model
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)


import matplotlib.pyplot as plt
from matplotlib import cm
pred = model.predict(np.reshape(train_x[18,:,:,:], (1, 128, 256, 1)))
plt.imshow(np.reshape(pred, (128, 256)), cmap=cm.Reds)
pred.shape
plt.imshow(np.reshape(train_y[18], (128, 256)), cmap=cm.Reds)
from keras.models import load_model
model.save('steady_flow_cnn_2_4k_train.h5')