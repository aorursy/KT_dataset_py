# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout,Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

%load_ext tensorboard



# Helper libraries

import numpy as np

import random

import matplotlib.pyplot as plt

import datetime



print(tf.__version__)
#set the seed

from numpy.random import seed

seed(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
fashion_mnist = keras.datasets.fashion_mnist



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#create an array to access the class name based on label number.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(len(train_images),len(test_images))
#Check shape of training image

train_images[0].shape
plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
plt.figure(figsize=(15,15))

for i in range(25):

  plt.subplot(5,5,i+1)

  plt.xticks([])

  plt.yticks([])

  rand_no = random.randint(0,len(train_images))     

  plt.imshow(train_images[rand_no], cmap='gray')

  plt.xlabel(class_names[train_labels[rand_no]])
#Normalizing the pixel values

train_images = train_images / 255.0

test_images = test_images / 255.0
# Clear any logs from previous runs

!rm -rf ./logs/ 



def create_model():

  model = Sequential()

  #input layer size is 784 after flattening

  model.add(Flatten(input_shape=(28, 28)))

  #hidden layer with 512 neurons

  model.add(Dense(512, activation='relu'))

  model.add(Dense(10, activation='softmax'))

  return model
model = create_model()

model.summary()

model.compile(optimizer='adam',

              loss=SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

earlystopping_callback = EarlyStopping(

    monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',

    baseline=None, restore_best_weights=True

)

#modelcheckpoint_callback = ModelCheckPoint('best_model.hdf5',monitor='val_loss', verbose=0, mode='auto',save_best_only=True)
model.fit(x=train_images, 

          y=train_labels, 

          epochs=20, 

          validation_split=0.2, 

          callbacks=[tensorboard_callback,earlystopping_callback])
!kill 239
%tensorboard --logdir logs/fit
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



print('\nTest accuracy:', test_acc)
predictions = model.predict(test_images)
predictions.shape
class_names[np.argmax(predictions[189])]
plt.imshow(test_images[189],cmap='gray')
def plot_image(i, predictions_array, true_label, img):

  true_label, img = true_label[i], img[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])



  plt.imshow(img, cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:

    color = 'blue'

  else:

    color = 'red'



  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)
# Plot the first X test images, their predicted labels, and the true labels.

# Color correct predictions in blue and incorrect predictions in red.

num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(i, predictions[i], test_labels, test_images)

plt.tight_layout()

plt.show()
# Save the entire model as a SavedModel.

!mkdir -p saved_model

model.save('saved_model/my_model')
#Loading the model from saved location

loaded_model = tf.keras.models.load_model('saved_model/my_model')



# Check its architecture

loaded_model.summary()