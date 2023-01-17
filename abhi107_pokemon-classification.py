
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mig

import pathlib
PATH = '../input/pokemonclassification/PokemonData/'
data_dir = pathlib.Path(PATH)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES

from tensorflow.keras.preprocessing.image import ImageDataGenerator
IDG = ImageDataGenerator(rescale = 1./255, validation_split=0.2, )

train_data = IDG.flow_from_directory(PATH,target_size=(224,224),batch_size=32,classes = list(CLASS_NAMES),subset='training')
validation_data = IDG.flow_from_directory(PATH,target_size=(224,224),batch_size=32,classes = list(CLASS_NAMES),subset='validation')

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n].argmax()])
      plt.axis('off')
     
image_batch, label_batch = next(train_data)
show_batch(image_batch, label_batch)
from keras.models import Sequential
from keras import layers
from keras.layers import BatchNormalization
from keras import regularizers
model = Sequential()
##Convutional Layers
model.add(layers.Conv2D(8, (4, 4),input_shape=(224,224,3)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(16, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

##Fully Conneted Layers

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(.5))
model.add(layers.Dense(len(CLASS_NAMES),activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

hist = model.fit_generator(train_data, epochs=37, steps_per_epoch = train_data.samples//32, validation_data=validation_data, validation_steps = validation_data.samples//32)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['accuracy'],label='accuracy',color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, step=0.04))
plt.show()
plt.figure(figsize=(20,20))
#for _ in range(3):
sam_x,sam_y = next(validation_data) 
pred_ = model.predict(sam_x)
for i in range(15):
    pred,y = pred_[i].argmax(), sam_y[i].argmax()
    plt.subplot(4,4,i+1)
    plt.imshow(sam_x[i])
    title_ = 'Predict:' + str(CLASS_NAMES[pred])+ ';   Label:' + str(CLASS_NAMES[y])
    plt.title(title_,size=11)
plt.show()