import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import plot_model
print(os.listdir('../input/cell-images-for-detecting-malaria'))
base_dir = '../input/cell-images-for-detecting-malaria/cell_images'
parasite_dir = os.path.join(base_dir,'Parasitized')
uninfect_dir = os.path.join(base_dir, 'Uninfected')
print(f'Length of Parasites is {len(os.listdir(parasite_dir))}')
print(f'Length of Uninfected is {len(os.listdir(uninfect_dir))}')
parasite_images = os.listdir(parasite_dir)
uninfected_images = os.listdir(uninfect_dir)
plt.imshow(mpimg.imread(os.path.join(parasite_dir,parasite_images[10])))
plt.title('Parasite Image')
plt.imshow(mpimg.imread(os.path.join(uninfect_dir,uninfected_images[10])))
plt.title('Uninfected Image')
nrows = 4
ncols = 4
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_parasite_pix = [os.path.join(parasite_dir, fname) 
                for fname in parasite_images[pic_index-8:pic_index]]
next_huninfected_pix = [os.path.join(uninfect_dir, fname) 
                for fname in uninfected_images[pic_index-8:pic_index]]

for i, img_path in enumerate(next_parasite_pix+next_huninfected_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
data_gen = ImageDataGenerator(
    rescale=1./255,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    validation_split=0.2)
train_data_gen = data_gen.flow_from_directory(os.path.join(base_dir,'cell_images'),
                                              target_size=(64, 64),
                                             batch_size = 32,
                                             class_mode='binary',
                                             subset='training')
test_data_gen = data_gen.flow_from_directory(os.path.join(base_dir,'cell_images'),
                                              target_size=(64, 64),
                                             batch_size = 32,
                                             class_mode='binary',
                                             subset='validation')
model = Sequential([
    Conv2D(16, (3, 3), activation = 'relu', input_shape = (64, 64,3)),
    MaxPool2D(2, 2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Dropout(0.1),
#     Conv2D(512, (3,3), activation='relu'),
#     MaxPool2D(2,2),
#     Dropout(0.1),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
    
])
model.summary()
plot_model(model)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
history = model.fit(train_data_gen, steps_per_epoch = 100, epochs= 30, validation_data = test_data_gen,
         validation_steps = 10, verbose = 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
