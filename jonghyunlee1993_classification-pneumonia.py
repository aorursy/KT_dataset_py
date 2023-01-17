import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
import glob

data_path = '/kaggle/input/chest-xray-pneumonia/chest_xray' 

train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')
val_data_path = os.path.join(data_path, 'val')
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

CLASS_NAMES = ['NORMAL', 'PNEMONIA']

train_datagen = ImageDataGenerator(
    rescale=1./ 255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10,
    )

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(226,226),
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    subset='training'
  )

validation_generator = train_datagen.flow_from_directory(train_data_path, 
                                                         target_size=(226,226), 
                                                         batch_size=32,  
                                                         class_mode='binary',
                                                         classes=CLASS_NAMES,
                                                         subset='validation')
import matplotlib.pyplot as plt
%matplotlib inline

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[int(label_batch[n])])
      plt.axis('off')
        
image_batch, label_batch = validation_generator[6]
show_batch(image_batch, label_batch)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(226, 226, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
IDG_test = ImageDataGenerator(rescale = 1./255)
test_data = IDG_test.flow_from_directory(test_data_path, 
                                         target_size=(226, 226),
                                         shuffle=False,
                                         batch_size=32,
                                         class_mode='binary',
                                         classes = CLASS_NAMES)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=7,
    validation_data=validation_generator,
   )
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
print(model.metrics_names)

loss, accuracy = model.evaluate(test_data)

print(f"Loss: {loss}\nAccuracy: {accuracy}")
