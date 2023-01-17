import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
paper_path = '/kaggle/input/rockpaperscissors/paper/'
rock_path = '/kaggle/input/rockpaperscissors/rock/'
scissors_path = '/kaggle/input/rockpaperscissors/scissors/'
data_paper = sum(os.path.isfile(os.path.join(paper_path, f))
                 for f in os.listdir(paper_path))
data_rock = sum(os.path.isfile(os.path.join(rock_path, f))
                 for f in os.listdir(rock_path))
data_scissors = sum(os.path.isfile(os.path.join(scissors_path, f))
                 for f in os.listdir(scissors_path))
counts = [data_paper, data_rock, data_scissors]
groups = ['paper\n' + str(counts[0]),
          'rock\n' + str(counts[1]),
          'scissors\n' + str(counts[2])]

colors = ['b', 'g', 'r']
plt.title('Amount of data')

width = len(counts) * 0.2
plt.bar(groups, counts, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)
# The data directory for training
train_dir = '/kaggle/working/train_dir'
# The data directory for validation
val_dir = '/kaggle/working/val_dir'
# The data directory for testing
test_dir = '/kaggle/working/test_dir'
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "rock"))
    os.makedirs(os.path.join(dir_name, "paper"))
    os.makedirs(os.path.join(dir_name, "scissors"))
create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)
paths = {
    'paper': [],
    'rock': [],
    'scissors': []
}
for dirname, _, filenames in os.walk(paper_path):
    for filename in filenames:
        paths['paper'].append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk(rock_path):
    for filename in filenames:
        paths['rock'].append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk(scissors_path):
    for filename in filenames:
        paths['scissors'].append(os.path.join(dirname, filename))
def copy_images(start_index, end_index, paths, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(paths[i],
                    os.path.join(dest_dir, paths[i].split('/')[4]))
# Part of the test data set
test_val_portion = 0.15
for value in paths.values():
    copy_images(0, len(value) - int(len(value) * 2 * test_val_portion),
                value, train_dir)
    copy_images(len(value) - int(len(value) * 2 * test_val_portion), len(value) - int(len(value) * test_val_portion),
                value, val_dir)
    copy_images(len(value) - int(len(value) * test_val_portion), len(value),
                value, test_dir)
target_size = (60, 60)
batch_size = 32
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
test_gen = datagen.flow_from_directory(test_dir,
                                        target_size=target_size,
                                        batch_size=batch_size,
                                        class_mode=mode)
len_train_data = len(train_gen.filenames)
len_test_data = len(test_gen.filenames)
len_val_data = len(val_gen.filenames)
labels = ['Train data', 'Test data', 'Validation data']
values = [len_train_data, len_test_data, len_val_data]
colors = ['green','red','blue']
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
model.add(Conv2D(64, (3, 3), input_shape=(60, 60, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics='accuracy')
history = model.fit_generator(train_gen,
                   steps_per_epoch=len_train_data // batch_size,
                   epochs=25,
                   validation_data=val_gen,
                   validation_steps=len_val_data // batch_size,
                   verbose=1,
                   callbacks=callbacks_list)
model.load_weights(best_model_path)
testing_model = model.evaluate_generator(test_gen, len_test_data // batch_size, verbose=1)
print('Percentage of correct responses: ' + str(int(testing_model[1] * 10000) / 100) + '%')
plt.plot(history.history['accuracy'], 
         label='Accuracy')
plt.plot(history.history['val_accuracy'],
         label='Validation accuracy')
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
counts = [testing_model[0], testing_model[1]]
groups = ['Testing model loss\n' + str(int(counts[0] * 10000) / 100) + '%',
          'Testing model accuracy\n' + str(int(counts[1] * 10000) / 100) + '%']

colors = ['r', 'b']
plt.title('Amount of data')

width = len(counts) * 0.3
plt.bar(groups, counts, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)