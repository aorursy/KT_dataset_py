from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data_dir = '/kaggle/input/flowers-recognition/flowers/'
flowers = {
    'sunflower': [],
    'tulip': [],
    'daisy': [],
    'rose': [],
    'dandelion': []}
for flower in flowers.keys():
    for dirname, _, filenames in os.walk(os.path.join(data_dir, flower)):
        for filename in filenames:
            flowers[flower].append((os.path.join(
                os.path.join(data_dir, flower), filename)))
groups = [flower + '\n' + str(len(flowers[flower])) 
          for flower in flowers]
count_data_flowers = [len(flowers[flower])
                     for flower in flowers]

colors = ['b', 'g', 'r', 'yellow', 'orange']
plt.title('Amount of data')

width = len(count_data_flowers) * 0.1
plt.bar(groups, count_data_flowers, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)
for i in range(len(flowers)):
    fig, ax = plt.subplots()
    img_path = flowers[list(flowers.keys())[i]][0]
    img = image.load_img(img_path, target_size=(150, 150))
    ax.imshow(img)
    plt.title(list(flowers.keys())[i])
    
    fig.set_figwidth(5)
    fig.set_figheight(5)
plt.show()
train_dir = '/kaggle/working/train_dir'
val_dir = '/kaggle/working/val_dir'
test_dir = '/kaggle/working/test_dir'
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    
    for flower in flowers.keys():
        os.makedirs(os.path.join(dir_name, flower))
create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)
def copy_images(start_index, end_index, paths, dest_dir):
    flower = paths[0].split('/')[5]
    for i in range(start_index, end_index):
        dest_path = os.path.join(dest_dir, flower)
        shutil.copy2(paths[i], dest_path)
# Part of the validation and test data set
val_test_data_proportion = 0.15
# copying images from input directory to output
for paths in flowers.values():
    copy_images(0, len(paths) - int(val_test_data_proportion * 2 * len(paths)),
                paths, train_dir)
    copy_images(len(paths) - int(val_test_data_proportion * 2 * len(paths)),
                len(paths) - int(val_test_data_proportion * len(paths)), paths, val_dir)
    copy_images(len(paths) - int(val_test_data_proportion * len(paths)),
                len(paths), paths, test_dir)
target_size = (90, 90)
batch_size = 50
mode = 'categorical'
datagen = ImageDataGenerator(rescale=1. / 255,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
train_gen = datagen.flow_from_directory(train_dir,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       class_mode=mode)
val_gen = datagen.flow_from_directory(val_dir,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       class_mode=mode)
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_gen = test_datagen.flow_from_directory(test_dir,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       class_mode=mode)
# indexes of labels
train_gen.class_indices
len_train_data = len(train_gen.filenames)
len_test_data = len(test_gen.filenames)
len_val_data = len(val_gen.filenames)
labels = ['Train data', 'Test data', 'Validation data']
values = [len_train_data, len_test_data, len_val_data]
colors = ['green','orange','blue']
plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, explode=(0, 0.1, 0.1))
plt.axis('equal')
plt.title('Amount of data')
plt.show()
best_model_path = 'best_model.h5'
checkpoint_callback = ModelCheckpoint(best_model_path,
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     verbose=1)
reduce_callback = ReduceLROnPlateau(monitor='val_accuracy',
                                   patience=3,
                                   factor=0.5,
                                   min_lr=0.00001,
                                   verbose=1)
callbacks_list = [checkpoint_callback, reduce_callback]
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(90, 90, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer='RMSProp',
             metrics=['accuracy'])
history = model.fit_generator(train_gen,
                   steps_per_epoch=len_train_data // batch_size,
                   epochs=50,
                   validation_data=val_gen,
                   validation_steps=len_val_data // batch_size,
                   verbose=1,
                   callbacks=callbacks_list)
model.load_weights(best_model_path)
results = model.evaluate_generator(test_gen, len_test_data // batch_size, verbose=1)
plt.plot(history.history['accuracy'], 
         label='Accuracy')
plt.plot(history.history['val_accuracy'],
         label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Percentage of correct responses')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 
         label='Loss')
plt.plot(history.history['val_loss'],
         label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Percentage of loss')
plt.legend()
plt.show()