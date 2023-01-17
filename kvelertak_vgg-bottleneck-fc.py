# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# !mkdir dataset
# !mkdir dataset/dogs
# !mkdir dataset/cats
!ls -ltr dataset
import numpy as np
import keras
from keras.models import Model
from keras.applications import mobilenet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
import re
from random import shuffle
from glob import glob
import shutil  as sh
%matplotlib inline
from matplotlib import pyplot as plt
IMG_SIZE = (224, 224)  # размер входного изображения сети
train_files = glob('../input/train/*.jpg')
test_files = glob('../input/test/*.jpg')
#Only once for creating train dir - uncomment to execute

# dataset_dir = 'dataset'
# for path  in train_files:
#     fname =  os.path.basename(path)
# #     print(os.path.join(dataset_dir,'dogs',fname))
# #     break
    
#     if re.match('.*/dog\.\d', path):        
#         sh.copyfile(path, os.path.join(dataset_dir,'dogs',fname))
#     else:
#         sh.copyfile(path, os.path.join(dataset_dir,'cats',fname))
train_dir = 'dataset/'
test_dir = '../input/test'   
# !ls -ltr dataset/cats

# #save_bottlebeck_features()
# batch_size=8
# datagen = ImageDataGenerator(rescale=1. / 255,
#                                  validation_split = 0.12,
#                                  preprocessing_function = mobilenet.preprocess_input)

# # build the VGG16 network
# model = VGG16(include_top = False,
#                    weights = 'imagenet',
#                    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

# generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False,
#     subset = 'training')
# nb_train_samples = generator.n
# bottleneck_features_train = model.predict_generator(
#     generator, nb_train_samples // batch_size , max_queue_size=500)
# np.save(open('bottleneck_features_train.npy', 'wb'),
#         bottleneck_features_train)

# generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False,
#     subset = 'validation')
# nb_validation_samples = generator.n
# bottleneck_features_validation = model.predict_generator(
#     generator, nb_validation_samples // batch_size , max_queue_size=500)
# np.save(open('bottleneck_features_validation.npy', 'wb'),
#         bottleneck_features_validation)
# batch_size=8
# datagen = ImageDataGenerator(rescale=1. / 255,
#                                  validation_split = 0.12,
#                                  preprocessing_function = preprocess_input)

# train_generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False,
#     subset = 'training')

# valid_generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False,
#     subset = 'validation')
# train_data = np.load(open('bottleneck_features_train.npy','rb'))
# # the features were saved in order, so recreating the labels is easy
# train_labels = train_generator.classes

# validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
# validation_labels = valid_generator.classes
# fig = plt.figure(figsize=(20, 20))
# for i, path in enumerate(train_data[:10], 1):
#     subplot = fig.add_subplot(i // 5 + 1, 5, i)
#     plt.imshow(plt.imread(path));
#     subplot.set_title('%d', train_labels[i]);
# train_data.shape, train_labels.shape, validation_data.shape, validation_labels.shape
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(                            
                            rotation_range=40,
                            width_shift_range=0.3,
                            height_shift_range=0.3,
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.3,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            validation_split = 0.12,
                            preprocessing_function = preprocess_input)

# this is the augmentation configuration we will use for testing:
# only rescaling
predict_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=IMG_SIZE,  # all images will be resized to 150x150
        shuffle=True, 
        seed=13,
        batch_size=batch_size,
        class_mode='binary',
        subset = 'training')  
# this is a similar generator, for validation data
validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        shuffle=True, 
        seed=13,
        batch_size=batch_size,
        class_mode='binary',
        subset="validation")
base_model = VGG16(include_top = False,
                   weights = 'imagenet',
                   input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))
# фиксируем все веса предобученной сети
for layer in base_model.layers:
    layer.trainable = False
# base_model.summary()
# define the checkpoint
bottleneck_filepath = 'bottleneck_fc_model_1.h5'
checkpoint = ModelCheckpoint(bottleneck_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
x = base_model.layers[-1].output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(2048, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-5))(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(1024, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-5))(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(1,  # один выход
                activation='sigmoid',  # функция активации  
                kernel_regularizer=keras.regularizers.l2(1e-4))(x)
topLayersModel = Model(inputs=base_model.input, outputs=x)
# topLayersModel.summary()
topLayersModel.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# topLayersModel.fit_generator(train_generator,  # данные читаем функцией-генератором
#                     steps_per_epoch=22500 // batch_size , # число вызовов генератора за эпоху
#                     epochs=10,  # число эпох обучения
#                     validation_data=validation_generator,
#                     validation_steps=2500 // batch_size,
#                     callbacks=callbacks_list)
topLayersModel= keras.models.load_model('bottleneck_fc_model_1.h5')
# загружаем входное изображение и предобрабатываем
def load_image(path, target_size=IMG_SIZE):
    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения
    array = img_to_array(img)
    return preprocess_input(array)  # предобработка для VGG16
# генератор последовательного чтения тестовых данных с диска
def test_generator(files):
    while True:
        for path in files:
            yield np.array([load_image(path)])
pred = topLayersModel.predict_generator(test_generator(test_files), len(test_files), max_queue_size=500)
%matplotlib inline
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(20, 20))
for i, (path, score) in enumerate(zip(test_files[90:][:10], pred[90:][:10]), 1):
    subplot = fig.add_subplot(i // 5 + 1, 5, i)
    plt.imshow(plt.imread(path));
    subplot.set_title('%.3f' % score);
with open('submit_vgg_bottleneck.txt', 'w') as dst:
    dst.write('id,label\n')
    for path, score in zip(test_files, pred):
        dst.write('%s,%f\n' % (re.search('(\d+)', path).group(0), score))

