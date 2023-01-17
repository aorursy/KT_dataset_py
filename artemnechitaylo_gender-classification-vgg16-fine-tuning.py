from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train_data_dir = '/kaggle/input/gender-classification-dataset/Training'
val_data_dir = '/kaggle/input/gender-classification-dataset/Validation'
paths_dict = {
    'female': [],
    'male': []
}
for key in paths_dict.keys():
    for dirname, _, filenames in os.walk(os.path.join(train_data_dir, key)):
        for filename in filenames:
            paths_dict[key].append(os.path.join(dirname, filename))
groups = [key + '\n' + str(len(paths_dict[key])) 
          for key in paths_dict.keys()]
count_data = [len(paths_dict[key])
          for key in paths_dict.keys()]

colors = ['b', 'r']
plt.title('Amount of train data')

width = len(count_data) * 0.3
plt.bar(groups, count_data, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)
train_dir = '/kaggle/working/train_dir'
test_dir = '/kaggle/working/test_dir'
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    
    for key in paths_dict.keys():
        os.makedirs(os.path.join(dir_name, key))
create_directory(train_dir)
create_directory(test_dir)
def copy_images(start_index, end_index, paths, dest_dir):
    for i in range(start_index, end_index):
        dest_path = os.path.join(dest_dir, paths[i].split('/')[5])
        shutil.copy2(paths[i], dest_path)
# Part of the test data set
test_data_proportion = 0.2
for key in paths_dict.keys():
    test_index = len(paths_dict[key]) - int(len(paths_dict[key]) * test_data_proportion)
    
    copy_images(0, test_index, paths_dict[key], train_dir)
    copy_images(test_index, len(paths_dict[key]), paths_dict[key], test_dir)
target_size = (224, 224)
batch_size = 30
mode = 'binary'
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
val_gen = datagen.flow_from_directory(val_data_dir,
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
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.trainable = False
model_without_tuning = 'best_model.h5'
checkpoint_callback = ModelCheckpoint(model_without_tuning,
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     verbose=1)
model = Sequential()
model.add(vgg)

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             metrics=['accuracy'],
             optimizer='adam')
history_without_tuning = model.fit_generator(train_gen,
                   steps_per_epoch=len_train_data // batch_size,
                   epochs=3,
                   validation_data=val_gen,
                   validation_steps=len_val_data // batch_size,
                   verbose=1,
                   callbacks=[checkpoint_callback])
model.load_weights(model_without_tuning)
results_without_tuning = model.evaluate_generator(test_gen,
                                   len_test_data // batch_size,
                                   verbose=1)
model.summary()
model.layers[0].trainable = True
for layer in model.layers[0].layers:
    if 'block3' not in layer.name:
        layer.trainable = False
model.summary()
model.compile(optimizer=Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model_with_tuning = 'best_tune_model.h5'
checkpoint_callback = ModelCheckpoint(model_with_tuning,
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     verbose=1)
history_with_tuning = model.fit_generator(train_gen,
                   steps_per_epoch=len_train_data // batch_size,
                   epochs=3,
                   validation_data=val_gen,
                   validation_steps=len_val_data // batch_size,
                   verbose=1,
                   callbacks=[checkpoint_callback])
model.load_weights(model_with_tuning)
results_with_tuning = model.evaluate_generator(test_gen,
                                   len_test_data // batch_size,
                                   verbose=1)
plt.plot(history_without_tuning.history['accuracy'], 
         label='Accuracy without tuning')
plt.plot(history_with_tuning.history['accuracy'],
         label='Accuracy with tuning')
plt.xlabel('Epoch')
plt.ylabel('Percentage of correct responses')
plt.title('Comparing accuracy')
plt.legend()
plt.show()
plt.plot(history_without_tuning.history['val_accuracy'], 
         label='Accuracy without tuning')
plt.plot(history_with_tuning.history['val_accuracy'],
         label='Accuracy with tuning')
plt.xlabel('Epoch')
plt.ylabel('Percentage of correct responses')
plt.title('Comparing val accuracy')
plt.legend()
plt.show()
counts = [results_without_tuning[1], results_with_tuning[1]]
groups = ['Without tuning\n' + str(int(counts[0] * 10000) / 100) + '%',
          'With tuning\n' + str(int(counts[1] * 10000) / 100) + '%']

colors = ['r', 'b']
plt.title('Accuracy based on test data')

width = len(counts) * 0.3
plt.bar(groups, counts, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)
counts = [results_without_tuning[0], results_with_tuning[0]]
groups = ['Without tuning\n' + str(int(counts[0] * 10000) / 100) + '%',
          'With tuning\n' + str(int(counts[1] * 10000) / 100) + '%']

colors = ['r', 'b']
plt.title('Loss based on test data')

width = len(counts) * 0.3
plt.bar(groups, counts, width=width, color=colors, alpha=0.6, bottom=2, linewidth=2)
img = image.load_img(paths_dict['female'][0], target_size=(224, 224))
plt.imshow(img)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.
model.layers[0].summary()
activation_model = Model(inputs=model.layers[0].layers[0].input, outputs=model.layers[0].layers[3].output)
activation = activation_model.predict(img_array)
images_per_row = 16
n_filters = activation.shape[-1]
size = activation.shape[1]
n_cols = n_filters // images_per_row
display_grid = np.zeros((n_cols * size, images_per_row * size))
for col in range(n_cols):
    for row in range(images_per_row):
        channel_image = activation[0, :, :, col * images_per_row + row]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')
activation_model = Model(inputs=model.layers[0].layers[0].input, outputs=model.layers[0].layers[6].output)
activation = activation_model.predict(img_array)
images_per_row = 16
n_filters = activation.shape[-1]
size = activation.shape[1]
n_cols = n_filters // images_per_row
display_grid = np.zeros((n_cols * size, images_per_row * size))
for col in range(n_cols):
    for row in range(images_per_row):
        channel_image = activation[0, :, :, col * images_per_row + row]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')