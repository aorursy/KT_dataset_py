import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import random
from itertools import chain
 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
#from tensorflow.keras.utils.plot_model import plot_model
import tensorflow as tf
 
%matplotlib inline
dog_breeds = os.listdir('../input/stanford-dogs-dataset/images/Images/')
print(dog_breeds)
filtered_breeds = [breed.split('-',1)[1] for breed in dog_breeds]
filtered_breeds[:]
X = []
y = []

fullpaths = ['../input/stanford-dogs-dataset/images/Images/{}'.format(dog_breeds) for dog_breeds in dog_breeds]

for counter, fullpath in enumerate(fullpaths):
    for imgname in os.listdir(fullpath):
        X.append([fullpath + '/' + imgname])
        y.append(filtered_breeds[counter])

X = list(chain.from_iterable(X))

len(X)
combined = list(zip(X, y))
random.shuffle(combined)

X[:], y[:] = zip(*combined)
X = X[:4000]
y = y[:4000]
labels = LabelEncoder()
labels.fit(y)
label_encoded = to_categorical(labels.transform(y), len(filtered_breeds))
label_encoded = np.array(label_encoded)
images = np.array([img_to_array(load_img(img, target_size = (299,299))) for img in X]) 

x_train, x_test, y_train, y_test = train_test_split(images, label_encoded, test_size = 0.3,stratify = np.array(y), random_state = 120) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size = 0.3,stratify=np.array(y_train),random_state = 120)

print('Training Dataset Size: ', x_train.shape)
print('Training Label Size: ', y_train.shape)
print('Validation Dataset Size: ', x_val.shape)
print('Validation Label Size: ', y_val.shape)
print('Testing Dataset Size: ', x_test.shape)
print('Testing Label Size: ', y_test.shape)
del images
gc.collect()
pre_trained_model =  InceptionV3(weights = "imagenet", input_shape=(299,299,3), include_top= False)

for layer in pre_trained_model.layers:
    layer.trainable=False
pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed9')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True
model = models.Sequential()
model.add(pre_trained_model)
model.add(GlobalAveragePooling2D())
model.add(Flatten()) 
model.add(Dropout(0.3))
model.add(Dense(2048, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(len(filtered_breeds), activation = 'softmax'))

model.compile(optimizer = RMSprop(lr=0.001), loss ='categorical_crossentropy', metrics =['accuracy'])

model.summary()
tf.keras.utils.plot_model(model, to_file='RMSprop.png', show_shapes=True, show_layer_names=True)
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
)

train_generator = train_datagen.flow(x_train, y_train,shuffle = False, batch_size = 25)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator =  val_datagen.flow(x_val, y_val, shuffle = False, batch_size = 25)
callbacks = myCallback()

epochs = 40
history = model.fit_generator(train_generator,epochs=epochs,validation_data=validation_generator,callbacks=[callbacks])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()