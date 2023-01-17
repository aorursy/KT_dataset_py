import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from __future__ import print_function
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
import os
os.listdir('../input/intel-image-classification/seg_train/seg_train/')
import matplotlib.pyplot as plt
import cv2
img = cv2.imread(r'/kaggle/input/intel-image-classification/seg_test/seg_test/sea/21568.jpg')
plt.imshow(img)
from sklearn.utils import shuffle
def get_images(directory):
    Images = []
    Labels = []
    
    label = 0

    for labels in os.listdir(directory):
        
        if labels == 'buildings':
            label = 0
            
        elif labels == 'forest':
            label = 1
            
        elif labels == 'glacier':
            label = 2
            
        elif labels == 'mountain':
            label = 3
            
        elif labels == 'sea':
            label = 4
            
        elif labels == 'street':
            label = 5
            
        for image_file in os.listdir(directory+labels):
            image = cv2.imread(directory+labels+'/'+image_file)
            
            image = cv2.resize(image,(150,150))
            
            Images.append(image)
            Labels.append(label)
        
    return shuffle(Images,Labels,random_state=817328462)
    
    
def get_classlabels(class_code):
    labels = {0:'buildings',1:'forest',2:'glacier', 3:'mountain',4:'sea',5:'street'}
Images,Labels = get_images('../input/intel-image-classification/seg_train/seg_train/')
Images = np.array(Images) #converting the list of images to numpy array.
Labels = np.array(Labels)
print("Shape of Images:",Images.shape)
print("Shape of Labels:",Labels.shape)
from random import randint
f,ax = plt.subplots(5,5) 
f.subplots_adjust(0,0,3,3)
for i in range(0,5,1):
    for j in range(0,5,1):
        rnd_number = randint(0,len(Images))
        ax[i,j].imshow(Images[rnd_number])
        ax[i,j].set_title(get_classlabels(Labels[rnd_number]))
        ax[i,j].axis('off')
img_rows = 50
img_cols = 50
batch_size = 128

train_data_dir = '../input/intel-image-classification/seg_train/seg_train'

train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 30,
    horizontal_flip = True,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_rows,img_cols),
    batch_size = batch_size,
    class_mode = 'categorical'
)
validation_data_dir = '../input/intel-image-classification/seg_test/seg_test'

validation_datagen = ImageDataGenerator(
    rescale = 1/255
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_rows,img_cols),
    batch_size = batch_size,
    class_mode = 'categorical'
)

num_classes = 6



model = Sequential()

# First CONV-ReLU Layer
model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (img_rows, img_cols, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Second CONV-ReLU Layer
model.add(Conv2D(64, (3, 3), padding = "same", input_shape = (img_rows, img_cols, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Max Pooling with Dropout 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 3rd set of CONV-ReLU Layers
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 4th Set of CONV-ReLU Layers
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Max Pooling with Dropout 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 5th Set of CONV-ReLU Layers
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 6th Set of CONV-ReLU Layers
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Max Pooling with Dropout 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# First set of FC or Dense Layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Second set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Final Dense Layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())
from keras.optimizers import RMSprop, SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint('img_classification.h5',
                             monitor = 'val_loss',
                             mode = "min",
                             save_best_only = True,
                             verbose = 1
                            )
earlystop = EarlyStopping(monitor = 'val_loss',
                         min_delta = 0,
                          patience = 3,
                          restore_best_weights = True,
                         verbose = 1)



callbacks = [earlystop, checkpoint]

model.compile(loss= 'categorical_crossentropy',
             optimizer = Adam(lr = 0.0001),
             metrics = ['accuracy'])

nb_training_samples = 19548
nb_validation_samples = 990
epochs = 30

history = model.fit_generator(
    train_generator,
   # steps_per_epochs = nb_training_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size
)
history.history
import matplotlib.pyplot as plot
plot.plot(history.history['accuracy'])
plot.plot(history.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()
