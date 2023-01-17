%matplotlib inline
# Imports

import numpy as np 

import pandas as pd 

from glob import glob 

from skimage.io import imread 

import os

import shutil

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.nasnet import NASNetMobile

from keras.applications.xception import Xception

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Concatenate, GlobalMaxPooling2D

from keras.models import Model

from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam

!pip install --upgrade pip

!pip install livelossplot

from livelossplot import PlotLossesKeras
# Output files

TRAINING_LOGS_FILE = "training_logs.csv"

MODEL_SUMMARY_FILE = "model_summary.txt"

MODEL_FILE = "histopathologic_cancer_detector.h5"

TRAINING_PLOT_FILE = "training.png"

VALIDATION_PLOT_FILE = "validation.png"

ROC_PLOT_FILE = "roc.png"

KAGGLE_SUBMISSION_FILE = "kaggle_submission.csv"

INPUT_DIR = '../input/'
# Hyperparams

SAMPLE_COUNT = 85000

TRAINING_RATIO = 0.9

IMAGE_SIZE = 96

EPOCHS = 12

BATCH_SIZE = 216

VERBOSITY = 1

TESTING_BATCH_SIZE = 5000
# Data setup

training_dir = INPUT_DIR + 'train/'

data_frame = pd.DataFrame({'path': glob(os.path.join(training_dir,'*.tif'))})

data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split('.')[0]) 

labels = pd.read_csv(INPUT_DIR + 'train_labels.csv')

data_frame = data_frame.merge(labels, on = 'id')

negatives = data_frame[data_frame.label == 0].sample(SAMPLE_COUNT)

positives = data_frame[data_frame.label == 1].sample(SAMPLE_COUNT)

data_frame = pd.concat([negatives, positives]).reset_index()

data_frame = data_frame[['path', 'id', 'label']]

data_frame['image'] = data_frame['path'].map(imread)



training_path = '../training'

validation_path = '../validation'



for folder in [training_path, validation_path]:

    for subfolder in ['0', '1']:

        path = os.path.join(folder, subfolder)

        os.makedirs(path, exist_ok=True)



training, validation = train_test_split(data_frame, train_size=TRAINING_RATIO, stratify=data_frame['label'])



data_frame.set_index('id', inplace=True)



for images_and_path in [(training, training_path), (validation, validation_path)]:

    images = images_and_path[0]

    path = images_and_path[1]

    for image in images['id'].values:

        file_name = image + '.tif'

        label = str(data_frame.loc[image,'label'])

        destination = os.path.join(path, label, file_name)

        if not os.path.exists(destination):

            source = os.path.join(INPUT_DIR + 'train', file_name)

            shutil.copyfile(source, destination)
# Data augmentation

from imgaug import augmenters as iaa 

import imgaug as ia

from random import shuffle

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
def get_seq():

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            iaa.Flipud(0.2), # vertically flip 20% of all images

            sometimes(iaa.Affine(

                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)

                rotate=(-10, 10), # rotate by -45 to +45 degrees

                shear=(-5, 5), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0

                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels

                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),

                    ]),

                    iaa.Invert(0.01, per_channel=True), # invert color channels

                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.9, 1.1), per_channel=0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent=(-1, 0),

                            first=iaa.Multiply((0.9, 1.1), per_channel=True),

                            second=iaa.ContrastNormalization((0.9, 1.1))

                        )

                    ]),

                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                ],

                random_order=True

            )

        ],

        random_order=True

    )

    return seq
def chunker(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
training_data_generator = ImageDataGenerator(rescale=1./255,

                                             horizontal_flip=True,

                                             vertical_flip=True,

                                             rotation_range=90,

                                             zoom_range=0.2, 

                                             width_shift_range=0.1,

                                             height_shift_range=0.1,

                                             shear_range=0.05,

                                             channel_shift_range=0.1)
# Data generation

training_generator = training_data_generator.flow_from_directory(training_path,

                                                                 target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                                                 batch_size=BATCH_SIZE,

                                                                 class_mode='binary')

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,

                                                                              target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                                                              batch_size=BATCH_SIZE,

                                                                              class_mode='binary')
# Model (LB 0.9558)

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

inputs = Input(input_shape)



input_tensor = Input(shape=(96,96,3))

xception = Xception(include_top=False, input_shape=input_shape)  

nas_net = NASNetMobile(input_tensor=input_tensor , include_top=False, weights='imagenet')



outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(xception(inputs)),

                                GlobalAveragePooling2D()(nas_net(inputs))])

outputs = Dropout(0.5)(outputs)

outputs = Dense(1, activation='sigmoid')(outputs)



model = Model(inputs, outputs)

model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()



#  Training

history = model.fit_generator(training_generator,

                              steps_per_epoch=len(training_generator), 

                              validation_data=validation_generator,

                              validation_steps=len(validation_generator),

                              epochs=EPOCHS,

                              verbose=VERBOSITY,

                              callbacks=[PlotLossesKeras(),

                                         ModelCheckpoint(MODEL_FILE,

                                                         monitor='val_acc',

                                                         verbose=VERBOSITY,

                                                         save_best_only=True,

                                                         mode='max'),

                                         CSVLogger(TRAINING_LOGS_FILE,

                                                   append=False,

                                                   separator=';')])

model.load_weights(MODEL_FILE)
# Training plots

epochs = [i for i in range(1, len(history.history['loss'])+1)]



plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")

plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")

plt.legend(loc='best')

plt.title('training')

plt.xlabel('epoch')

plt.savefig(TRAINING_PLOT_FILE, bbox_inches='tight')

plt.show()



plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")

plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")

plt.legend(loc='best')

plt.title('validation')

plt.xlabel('epoch')

plt.savefig(VALIDATION_PLOT_FILE, bbox_inches='tight')

plt.show()
# ROC validation plot

roc_validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path,

                                                                                  target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                                                                  batch_size=BATCH_SIZE,

                                                                                  class_mode='binary',

                                                                                  shuffle=False)

predictions = model.predict_generator(roc_validation_generator, steps=len(roc_validation_generator), verbose=VERBOSITY)

false_positive_rate, true_positive_rate, threshold = roc_curve(roc_validation_generator.classes, predictions)

area_under_curve = auc(false_positive_rate, true_positive_rate)



plt.plot([0, 1], [0, 1], 'k--')

plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')

plt.show()
# Kaggle testing

testing_files = glob(os.path.join(INPUT_DIR+'test/','*.tif'))

submission = pd.DataFrame()

for index in range(0, len(testing_files), TESTING_BATCH_SIZE):

    data_frame = pd.DataFrame({'path': testing_files[index:index+TESTING_BATCH_SIZE]})

    data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split(".")[0])

    data_frame['image'] = data_frame['path'].map(imread)

    images = np.stack(data_frame.image, axis=0)

    predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]

    predictions = np.array(predicted_labels)

    data_frame['label'] = predictions

    submission = pd.concat([submission, data_frame[["id", "label"]]])

submission.to_csv(KAGGLE_SUBMISSION_FILE, index=False, header=True)