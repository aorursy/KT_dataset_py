# Imports for Deep Learning

from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D

from keras.models import Sequential, load_model, Model

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet101



# Ensure consistency across runs

from numpy.random import seed

import random

seed(2)

from tensorflow import set_random_seed

set_random_seed(2)



# Imports to view data

import cv2

from glob import glob



# Metrics

from sklearn.metrics import classification_report, confusion_matrix



# Visualization

from keras.utils import print_summary

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



# Utils

from pathlib import Path

import pandas as pd

import numpy as np

from os import getenv

import time

import itertools



# Image Preprocessing

from skimage.filters import sobel, scharr



import os
# Set global variables

TRAIN_DIR = '../input/significant-asl-sign-language-alphabet-dataset/significant-asl-alphabet-training-set/Training Set'

TEST_DIR = '../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test'

CUSTOM_TEST_DIR = '../input/asl-alphabet-test/asl-alphabet-test'



TARGET_SIZE = (224, 224)

TARGET_DIMS = (224, 224, 3) # add channel for RGB

N_CLASSES = 27

VALIDATION_SPLIT = 0.1

BATCH_SIZE = 64



# Model saving for easier local iterations

MODEL_DIR = '/kaggle/working'

MODEL_PATH = MODEL_DIR + '/cnn-model.h5'

MODEL_WEIGHTS_PATH = MODEL_DIR + '/cnn-model.weights.h5'

MODEL_SAVE_TO_DISK = True



print('Save model to disk? {}'.format('Yes' if MODEL_SAVE_TO_DISK else 'No'))
CLASSES = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]

CLASSES.sort()



print('Following classes were found:')

print(', '.join(CLASSES))
def plot_one_sample_of_each(base_path):

    cols = 5

    rows = int(np.ceil(len(CLASSES) / cols))

    fig = plt.figure(figsize=(16, 20))

    

    for i in range(len(CLASSES)):

        cls = CLASSES[i]

        img_path = base_path + '/' + cls + '/**'

        path_contents = glob(img_path)

    

        imgs = random.sample(path_contents, 1)



        sp = plt.subplot(rows, cols, i + 1)

        plt.imshow(cv2.imread(imgs[0]))

        plt.title(cls)

        sp.axis('off')



    plt.show()

    return
plot_one_sample_of_each(TRAIN_DIR)
def preprocess_image(image):

    '''Function that will be implied on each input. The function

    will run after the image is resized and augmented.

    The function should take one argument: one image (Numpy tensor

    with rank 3), and should output a Numpy tensor with the same

    shape.'''

    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    return sobely



def make_generator(options):

    '''Creates two generators for dividing and preprocessing data.'''

    validation_split = options.get('validation_split', 0.0)

    preprocessor = options.get('preprocessor', None)

    data_dir = options.get('data_dir', TRAIN_DIR)



    augmentor_options = {

        'samplewise_center': True,

        'samplewise_std_normalization': True,

    }

    if validation_split is not None:

        augmentor_options['validation_split'] = validation_split

    

    if preprocessor is not None:

        augmentor_options['preprocessing_function'] = preprocessor

    

    flow_options = {

        'target_size': TARGET_SIZE,

        'batch_size': BATCH_SIZE,

        'shuffle': options.get('shuffle', None),

        'subset': options.get('subset', None),

    }



    data_augmentor = ImageDataGenerator(**augmentor_options)

    return data_augmentor.flow_from_directory(data_dir, **flow_options)
def load_model_from_disk():

    '''A convenience method for re-running certain parts of the

    analysis locally without refitting all the data.'''

    model_file = Path(MODEL_PATH)

    model_weights_file = Path(MODEL_WEIGHTS_PATH)

                      

    if model_file.is_file() and model_weights_file.is_file():

        print('Retrieving model from disk...')

        model = load_model(model_file.__str__())

                      

        print('Loading CNN model weights from disk...')

        model.load_weights(model_weights_file)

        return model

    

    return None



CNN_MODEL = load_model_from_disk()

REPROCESS_MODEL = (CNN_MODEL is None)



print('Need to reprocess? {}'.format(REPROCESS_MODEL))
def build_transfer_model(save=False):

    print('Building new ResNet101 model')

    

    base_model = ResNet101(weights='imagenet', include_top=False)

    

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    predictions = Dense(N_CLASSES, activation='softmax')(x)

    

    model = Model(inputs=base_model.input, outputs=predictions)

    

    # for layer in base_model.layers:

    #     layer.trainable = False

        

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    if save: model.save(MODEL_PATH)

        

    return model
def build_model(save=False):

    print('Building model afresh...')

    

    model = Sequential()

    

    model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=TARGET_DIMS))

    model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))

    model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))

    model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(N_CLASSES, activation='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if save: model.save(MODEL_PATH)

        

    return model



if REPROCESS_MODEL:

    CNN_MODEL = build_transfer_model(save=MODEL_SAVE_TO_DISK)



print_summary(CNN_MODEL)
def make_generator_for(subset):

    '''Create a generator for the training or validation set.'''

    generator_options = dict(

        validation_split=VALIDATION_SPLIT,

        shuffle=True,

        subset=subset,

        preprocessor=preprocess_image,

    )

    return make_generator(generator_options)





def fit_model(model, train_generator, val_generator, save=False):

    '''Fit the model with the training and validation generators.'''    

    history = model.fit_generator(train_generator, epochs=5, validation_data=val_generator, shuffle=True)

    

    if save: model.save_weights(MODEL_WEIGHTS_PATH)

    

    return history





CNN_TRAIN_GENERATOR = make_generator_for('training')

CNN_VAL_GENERATOR = make_generator_for('validation')



HISTORY = None

if REPROCESS_MODEL:

    start_time = time.time()

    HISTORY = fit_model(CNN_MODEL, CNN_TRAIN_GENERATOR, CNN_VAL_GENERATOR, save=MODEL_SAVE_TO_DISK)

    print('Fitting the model took ~{:.0f} second(s).'.format(time.time() - start_time))





columns=['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']

pd.DataFrame(data=[x.shape for x in CNN_MODEL.weights], columns=columns)
if HISTORY:

    print('Final Accuracy: {:.2f}%'.format(HISTORY.history['accuracy'][4] * 100))

    print('Validation set accuracy: {:.2f}%'.format(HISTORY.history['val_accuracy'][4] * 100))
def plot_confusion_matrix(cm, classes,

                      normalize=False,

                      title='Confusion matrix',

                      cmap=plt.cm.Blues):

    '''

    Plot a confusion matrix heatmap using matplotlib. This code was obtained from

    the scikit-learn documentation:



    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    '''

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return





def plot_confusion_matrix_with_default_options(y_pred, y_true, classes):

    '''Plot a confusion matrix heatmap with a default size and default options.'''

    cm = confusion_matrix(y_true, y_pred)

    with sns.axes_style('ticks'):

        plt.figure(figsize=(16, 16))

        plot_confusion_matrix(cm, classes)

        plt.show()

    return
def evaluate_model(generator):

    start_time = time.time()

    evaluations = CNN_MODEL.evaluate_generator(generator)

    for i in range(len(CNN_MODEL.metrics_names)):

        print("{}: {:.2f}%".format(

            CNN_MODEL.metrics_names[i], evaluations[i] * 100))

    print('Took {:.0f} seconds to evaluate this set.'.format(

        time.time() - start_time))



    start_time = time.time()

    predictions = CNN_MODEL.predict_generator(generator)

    print('Took {:.0f} seconds to get predictions on this set.'.format(

        time.time() - start_time))



    y_pred = np.argmax(predictions, axis=1)

    y_true = generator.classes

    return dict(y_pred=y_pred, y_true=y_true)





def evaluate_validation_dataset():

    gen_options = dict(

        validation_split=0.1,

        data_dir=TRAIN_DIR,

        shuffle=False,

        subset='validation',

        preprocessor=preprocess_image,

    )

    val_gen = make_generator(gen_options)

    return evaluate_model(val_gen)





def evaluate_test_dataset():

    gen_options = dict(

        validation_split=0.0,

        data_dir=CUSTOM_TEST_DIR,

        shuffle=False,

        preprocessor=preprocess_image,

    )

    test_gen = make_generator(gen_options)

    return evaluate_model(test_gen)
CNN_VALIDATION_SET_EVAL = evaluate_validation_dataset()

print(classification_report(**CNN_VALIDATION_SET_EVAL, target_names=CLASSES))
with sns.axes_style('ticks'):

    plot_confusion_matrix_with_default_options(**CNN_VALIDATION_SET_EVAL, classes=CLASSES)