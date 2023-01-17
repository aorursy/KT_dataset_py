import os

from glob import glob

import random

import time

import tensorflow

import datetime

os.environ['KERAS_BACKEND'] = 'tensorflow'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed



from tqdm import tqdm



import numpy as np

import pandas as pd

from IPython.display import FileLink

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns 

%matplotlib inline

from IPython.display import display, Image

import matplotlib.image as mpimg

import cv2



from sklearn.model_selection import train_test_split

from sklearn.datasets import load_files       

from keras.utils import np_utils

from sklearn.utils import shuffle

from sklearn.metrics import log_loss



from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.applications.vgg16 import VGG16
dataset = pd.read_csv('../input/driver_imgs_list.csv')

dataset.head(5)
by_drivers = dataset.groupby('subject')

unique_drivers = by_drivers.groups.keys()

print(unique_drivers)
# Load the dataset previously downloaded from Kaggle

NUMBER_CLASSES = 10

# Color type: 1 - grey, 3 - rgb



def get_cv2_image(path, img_rows, img_cols, color_type=3):

    # Loading as Grayscale image

    if color_type == 1:

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    elif color_type == 3:

        img = cv2.imread(path, cv2.IMREAD_COLOR)

    # Reduce size

    img = cv2.resize(img, (img_rows, img_cols)) 

    return img



# Training

def load_train(img_rows, img_cols, color_type=3):

    start_time = time.time()

    train_images = [] 

    train_labels = []

    # Loop over the training folder 

    for classed in tqdm(range(NUMBER_CLASSES)):

        print('Loading directory c{}'.format(classed))

        files = glob(os.path.join('..', 'input', 'train', 'c' + str(classed), '*.jpg'))

        for file in files:

            img = get_cv2_image(file, img_rows, img_cols, color_type)

            train_images.append(img)

            train_labels.append(classed)

    print("Data Loaded in {} second".format(time.time() - start_time))

    return train_images, train_labels 



def read_and_normalize_train_data(img_rows, img_cols, color_type):

    X, labels = load_train(img_rows, img_cols, color_type)

    y = np_utils.to_categorical(labels, 10)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    x_train = np.array(x_train, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)

    x_test = np.array(x_test, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)

    

    return x_train, x_test, y_train, y_test



# Validation

def load_test(size=200000, img_rows=64, img_cols=64, color_type=3):

    path = os.path.join('..', 'input', 'test', '*.jpg')

    files = sorted(glob(path))

    X_test, X_test_id = [], []

    total = 0

    files_size = len(files)

    for file in tqdm(files):

        if total >= size or total >= files_size:

            break

        file_base = os.path.basename(file)

        img = get_cv2_image(file, img_rows, img_cols, color_type)

        X_test.append(img)

        X_test_id.append(file_base)

        total += 1

    return X_test, X_test_id



def read_and_normalize_sampled_test_data(size, img_rows, img_cols, color_type=3):

    test_data, test_ids = load_test(size, img_rows, img_cols, color_type)

    

    test_data = np.array(test_data, dtype=np.uint8)

    test_data = test_data.reshape(-1,img_rows,img_cols,color_type)

    

    return test_data, test_ids
img_rows = 64

img_cols = 64

color_type = 1
x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols, color_type)

print('Train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')
nb_test_samples = 200

test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)

print('Test shape:', test_files.shape)

print(test_files.shape[0], 'Test samples')
# Statistics

# Load the list of names

names = [item[17:19] for item in sorted(glob("../input/train/*/"))]

test_files_size = len(np.array(glob(os.path.join('..', 'input', 'test', '*.jpg'))))

x_train_size = len(x_train)

categories_size = len(names)

x_test_size = len(x_test)

print('There are %s total images.\n' % (test_files_size + x_train_size + x_test_size))

print('There are %d training images.' % x_train_size)

print('There are %d total training categories.' % categories_size)

print('There are %d validation images.' % x_test_size)

print('There are %d test images.'% test_files_size)
# Plot figure size

plt.figure(figsize = (10,10))

# Count the number of images per category

sns.countplot(x = 'classname', data = dataset)

# Change the Axis names

plt.ylabel('Count')

plt.title('Categories Distribution')

# Show plot

plt.show()
# Find the frequency of images per driver

drivers_id = pd.DataFrame((dataset['subject'].value_counts()).reset_index())

drivers_id.columns = ['driver_id', 'Counts']

drivers_id
# Plotting class distribution

dataset['class_type'] = dataset['classname'].str.extract('(\d)',expand=False).astype(np.float)

plt.figure(figsize = (20,20))

dataset.hist('class_type', alpha=0.5, layout=(1,1), bins=10)

plt.title('Class distribution')

plt.show()
activity_map = {'c0': 'Safe driving', 

                'c1': 'Texting - right', 

                'c2': 'Talking on the phone - right', 

                'c3': 'Texting - left', 

                'c4': 'Talking on the phone - left', 

                'c5': 'Operating the radio', 

                'c6': 'Drinking', 

                'c7': 'Reaching behind', 

                'c8': 'Hair and makeup', 

                'c9': 'Talking to passenger'}
plt.figure(figsize = (12, 20))

image_count = 1

BASE_URL = '../input/train/'

for directory in os.listdir(BASE_URL):

    if directory[0] != '.':

        for i, file in enumerate(os.listdir(BASE_URL + directory)):

            if i == 1:

                break

            else:

                fig = plt.subplot(5, 2, image_count)

                image_count += 1

                image = mpimg.imread(BASE_URL + directory + '/' + file)

                plt.imshow(image)

                plt.title(activity_map[directory])
def create_submission(predictions, test_id, info):

    result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])

    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)

    

    now = datetime.datetime.now()

    

    if not os.path.isdir('kaggle_submissions'):

        os.mkdir('kaggle_submissions')



    suffix = "{}_{}".format(info,str(now.strftime("%Y-%m-%d-%H-%M")))

    sub_file = os.path.join('kaggle_submissions', 'submission_' + suffix + '.csv')

    

    result.to_csv(sub_file, index=False)

    

    return sub_file
batch_size = 40

nb_epoch = 10
!rm -f saved_models/weights_best_vanilla.hdf5
models_dir = "saved_models"

if not os.path.exists(models_dir):

    os.makedirs(models_dir)

    

checkpointer = ModelCheckpoint(filepath='saved_models/weights_best_vanilla.hdf5', 

                               monitor='val_loss', mode='min',

                               verbose=1, save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

callbacks = [checkpointer, es]
def create_model_v1():

    # Vanilla CNN model

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = 3, padding='same', activation = 'relu', input_shape=(img_rows, img_cols, color_type)))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Conv2D(filters = 128, padding='same', kernel_size = 3, activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Conv2D(filters = 256, padding='same', kernel_size = 3, activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Conv2D(filters = 512, padding='same', kernel_size = 3, activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Dropout(0.5))



    model.add(Flatten())



    model.add(Dense(500, activation = 'relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation = 'softmax'))

    

    return model
model_v1 = create_model_v1()



# More details about the layers

model_v1.summary()



# Compiling the model

model_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Training the Vanilla Model version 1

history_v1 = model_v1.fit(x_train, y_train, 

          validation_data=(x_test, y_test),

          callbacks=callbacks,

          epochs=nb_epoch, batch_size=batch_size, verbose=1)
model_v1.load_weights('saved_models/weights_best_vanilla.hdf5')
def plot_train_history(history):

    # Summarize history for accuracy

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    # Summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
plot_train_history(history_v1)
def plot_test_class(model, test_files, image_number, color_type=1):

    img_brute = test_files[image_number]

    img_brute = cv2.resize(img_brute,(img_rows,img_cols))

    plt.imshow(img_brute, cmap='gray')



    new_img = img_brute.reshape(-1,img_rows,img_cols,color_type)



    y_prediction = model.predict(new_img, batch_size=batch_size, verbose=1)

    print('Y prediction: {}'.format(y_prediction))

    print('Predicted: {}'.format(activity_map.get('c{}'.format(np.argmax(y_prediction)))))

    

    plt.show()
score = model_v1.evaluate(x_test, y_test, verbose=1)

print('Score: ', score)
plot_test_class(model_v1, test_files, 20)
!rm -f saved_models/weights_best_vanilla.hdf5
def create_model_v2():

    # Optimised Vanilla CNN model

    model = Sequential()



    ## CNN 1

    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_rows, img_cols, color_type)))

    model.add(BatchNormalization())

    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization(axis = 3))

    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Dropout(0.3))



    ## CNN 2

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization(axis = 3))

    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Dropout(0.3))



    ## CNN 3

    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization(axis = 3))

    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Dropout(0.5))



    ## Output

    model.add(Flatten())

    model.add(Dense(512,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(128,activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(10,activation='softmax'))



    return model
model_v2 = create_model_v2()



# More details about the layers

model_v2.summary()



# Compiling the model

model_v2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Training the Vanilla Model

history_v2 = model_v2.fit(x_train, y_train, 

          validation_data=(x_test, y_test),

          callbacks=callbacks,

          epochs=nb_epoch, batch_size=batch_size, verbose=1)
plot_train_history(history_v2)
model_v2.load_weights('saved_models/weights_best_vanilla.hdf5')
score = model_v2.evaluate(x_test, y_test, verbose=1)

print('Score: ', score)



y_pred = model_v2.predict(x_test, batch_size=batch_size, verbose=1)

score = log_loss(y_test, y_pred)

print('Score log loss:', score)
plot_test_class(model_v2, test_files, 101) # The model really performs badly
plot_test_class(model_v2, test_files, 1) # The model really performs badly
plot_test_class(model_v2, test_files, 143) 
!rm -f saved_models/weights_best_vanilla.hdf5
# Prepare data augmentation configuration

train_datagen = ImageDataGenerator(rescale = 1.0/255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True, 

                                   validation_split = 0.2)



test_datagen = ImageDataGenerator(rescale=1.0/ 255, validation_split = 0.2)
nb_train_samples = x_train.shape[0]

nb_validation_samples = x_test.shape[0]

print(nb_train_samples)

print(nb_validation_samples)

training_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

validation_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)
checkpoint = ModelCheckpoint('saved_models/weights_best_vanilla.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history_v3 = model_v2.fit_generator(training_generator,

                         steps_per_epoch = nb_train_samples // batch_size,

                         epochs = 5, 

                         callbacks=[es, checkpoint],

                         verbose = 1,

                         validation_data = validation_generator,

                         validation_steps = nb_validation_samples // batch_size)
model_v2.load_weights('saved_models/weights_best_vanilla.hdf5')
plot_train_history(history_v3)
# Evaluate the performance of the new model

score = model_v2.evaluate_generator(validation_generator, nb_validation_samples // batch_size)

print("Test Score:", score[0])

print("Test Accuracy:", score[1])
plot_test_class(model_v2, test_files, 101)
plot_test_class(model_v2, test_files, 1) 
plot_test_class(model_v2, test_files, 145) 
plot_test_class(model_v2, test_files, 143) 
predictions = model_v2.predict(test_files, batch_size=batch_size)

FileLink(create_submission(predictions, test_targets, score[0]))
!rm -f saved_models/weights_best_vanilla.hdf5
def vgg_std16_model(img_rows, img_cols, color_type=3):

    nb_classes = 10

    # Remove fully connected layer and replace

    # with softmax for classifying 10 classes

    vgg16_model = VGG16(weights="imagenet", include_top=False)



    # Freeze all layers of the pre-trained model

    for layer in vgg16_model.layers:

        layer.trainable = False

        

    x = vgg16_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(nb_classes, activation = 'softmax')(x)



    model = Model(input = vgg16_model.input, output = predictions)

    

    return model
# Load the VGG16 network

print("Loading network...")

model_vgg16 = vgg_std16_model(img_rows, img_cols)



model_vgg16.summary()



model_vgg16.compile(loss='categorical_crossentropy',

                         optimizer='rmsprop',

                         metrics=['accuracy'])
training_generator = train_datagen.flow_from_directory('../input/train', 

                                                 target_size = (img_rows, img_cols), 

                                                 batch_size = batch_size,

                                                 shuffle=True,

                                                 class_mode='categorical', subset="training")



validation_generator = test_datagen.flow_from_directory('../input/train', 

                                                   target_size = (img_rows, img_cols), 

                                                   batch_size = batch_size,

                                                   shuffle=False,

                                                   class_mode='categorical', subset="validation")

nb_train_samples = 17943

nb_validation_samples = 4481
!rm -f saved_models/weights_best_vgg16.hdf5
# Training the Vanilla Model

checkpoint = ModelCheckpoint('saved_models/weights_best_vgg16.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history_v4 = model_vgg16.fit_generator(training_generator,

                         steps_per_epoch = nb_train_samples // batch_size,

                         epochs = 5, 

                         callbacks=[es, checkpoint],

                         verbose = 1,

                         class_weight='auto',

                         validation_data = validation_generator,

                         validation_steps = nb_validation_samples // batch_size)
model_vgg16.load_weights('saved_models/weights_best_vgg16.hdf5')
plot_train_history(history_v4)
def plot_vgg16_test_class(model, test_files, image_number):

    img_brute = test_files[image_number]



    im = cv2.resize(cv2.cvtColor(img_brute, cv2.COLOR_BGR2RGB), (img_rows,img_cols)).astype(np.float32) / 255.0

    im = np.expand_dims(im, axis =0)



    img_display = cv2.resize(img_brute,(img_rows,img_cols))

    plt.imshow(img_display, cmap='gray')



    y_preds = model.predict(im, batch_size=batch_size, verbose=1)

    print(y_preds)

    y_prediction = np.argmax(y_preds)

    print('Y Prediction: {}'.format(y_prediction))

    print('Predicted as: {}'.format(activity_map.get('c{}'.format(y_prediction))))

    

    plt.show()
plot_vgg16_test_class(model_vgg16, test_files, 133) # Texting left
plot_vgg16_test_class(model_vgg16, test_files, 29) # Texting left
plot_vgg16_test_class(model_vgg16, test_files, 82) # Hair
# Evaluate the performance of the new model

score = model_vgg16.evaluate_generator(validation_generator, nb_validation_samples // batch_size, verbose = 1)

print("Test Score:", score[0])

print("Test Accuracy:", score[1])