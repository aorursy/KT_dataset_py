from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data_train_dir = '/kaggle/input/fingers/fingers/train/'
data_test_dir = '/kaggle/input/fingers/fingers/test/'
train_dir = '/kaggle/working/train_dir'
val_dir = '/kaggle/working/val_dir'
test_dir = '/kaggle/working/test_dir'
def create_directory(dir_name):
    dirs = [os.path.join(dir_name, "L"), os.path.join(dir_name, "R")]
    
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    
    for dir in dirs:
        for i in range(6):
            os.makedirs(dir + str(i))
create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)
paths_train = []
paths_test = []
count_data = {'0L': 0,
              '1L': 0,
              '2L': 0,
              '3L': 0,
              '4L': 0,
              '5L': 0,
              '0R': 0,
              '1R': 0,
              '2R': 0,
              '3R': 0,
              '4R': 0,
              '5R': 0}
for dirname, _, filenames in os.walk(data_train_dir):
    for filename in filenames:
        paths_train.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk(data_test_dir):
    for filename in filenames:
        paths_test.append(os.path.join(dirname, filename))
for filename in paths_train:
    for key in count_data.keys():
        if filename[-6:-4] == key:
            count_data[key] += 1
            
for filename in paths_test:
    for key in count_data.keys():
        if filename[-6:-4] == key:
            count_data[key] += 1
counts = count_data.values()
groups = count_data.keys()
plt.title('Amount of labels')

width = len(counts) * 0.2
plt.bar(groups, counts, width=width, alpha=0.6, bottom=2, linewidth=2)
def copy_images(start_index, end_index, paths, dest_dir):
    for i in range(start_index, end_index):
        dest_path = os.path.join(dest_dir, paths[i][-5] + paths[i][-6])
        shutil.copy2(paths[i], dest_path)
# Part of the validation data set
validation_data_proportion = 0.15
# copying images from input directory to output
copy_images(0, len(paths_train) - int(validation_data_proportion * len(paths_train)),
            paths_train, train_dir)
copy_images(len(paths_train) - int(validation_data_proportion * len(paths_train)),
            len(paths_train), paths_train, val_dir)
copy_images(0, len(paths_test), paths_test, test_dir)
target_size = (90, 90)
batch_size = 50
mode = 'categorical'
datagen = ImageDataGenerator(rescale=1. / 255)
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
# indexes of labels
train_gen.class_indices
len_train_data = len(train_gen.filenames)
len_test_data = len(test_gen.filenames)
len_val_data = len(val_gen.filenames)
labels = ['Train data', 'Test data', 'Validation data']
values = [len_train_data, len_test_data, len_val_data]
colors = ['green','red','blue']
plt.pie(values, labels=labels, colors=colors)
plt.axis('equal')
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
model.add(Conv2D(64, (3, 3), input_shape=(90, 90, 3), activation='relu'))
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
model.add(Dense(12, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer='RMSprop',
             metrics=['accuracy', 'AUC'])
history = model.fit_generator(train_gen,
                   steps_per_epoch=len_train_data // batch_size,
                   epochs=3,
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