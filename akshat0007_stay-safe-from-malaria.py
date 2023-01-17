from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

sns.set_style({'xtick.bottom':False,
               'ytick.left':False,
               'axes.spines.bottom': False,
               'axes.spines.left': False,
               'axes.spines.right': False,
               'axes.spines.top': False})
im_list = [162+i for i in range(9)]
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
for i in enumerate(im_list):
    img = plt.imread("../input/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_"+str(i[1])+".png")
    ax=axes[i[0]//3,i[0]%3]
    ax.imshow(img)           
datagen = ImageDataGenerator(rescale=1./255,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      validation_split=0.2
                                      )
train_data = datagen.flow_from_directory('../input/cell_images/cell_images',
                                                     target_size=(64,64),
                                                     batch_size=128,
                                                     class_mode = 'binary',
                                                     subset = 'training')

validation_data = datagen.flow_from_directory('../input/cell_images/cell_images',
                                                     target_size=(64,64),
                                                     batch_size=128,
                                                     class_mode = 'binary',
                                                     subset = 'validation')
import tensorflow as tf
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(64,64,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['acc'])

history = model.fit_generator(train_data,
            validation_data = validation_data,
            
            epochs = 20,
            
            verbose = 1)
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

