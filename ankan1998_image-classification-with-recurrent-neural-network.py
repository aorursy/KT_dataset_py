import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
print(tf.__version__)
tf.random.set_seed(99)
fas_mnist=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fas_mnist.load_data()
train_images=train_images.reshape(60000, 28, 28)

train_images=train_images / 255.0 #Standardising

test_images = test_images.reshape(10000, 28, 28)

test_images=test_images/255.0 #Standardising
train_images.shape
test_images.shape
# The concept is simple, we take each HxW matrix of images --> Flatten it like sequence of Multi-dimensional time-series and feed to LSTM

# HxW changes to TxD 

# In images H--> height, W--> width, similiarly T-->Timestamp(equals H), D-->Feature(equals W)

model = tf.keras.Sequential([

  tf.keras.Input(shape=(28,28)),

  tf.keras.layers.GRU(128),

  tf.keras.layers.Dense(128, activation='relu',input_shape=(28, 28, )),

  tf.keras.layers.Dropout(0.2,input_shape=(128,)),

  tf.keras.layers.Dense(10, activation='softmax')

])



model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
def scheduler(epoch, lr):

      if epoch < 8:

        return lr

      else:

        return lr * tf.math.exp(-0.1)
my_callbacks = [

    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2),

    tf.keras.callbacks.LearningRateScheduler(scheduler)

]
trainer=model.fit(train_images, train_labels,validation_data=(test_images,test_labels), epochs=20,callbacks=my_callbacks)
# Plot loss per iteration

plt.plot(trainer.history['loss'], label='loss')

plt.plot(trainer.history['val_loss'], label='val_loss')

plt.legend()
# Plot accuracy per iteration

plt.plot(trainer.history['accuracy'], label='acc')

plt.plot(trainer.history['val_accuracy'], label='val_acc')

plt.legend()