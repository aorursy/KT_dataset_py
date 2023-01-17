import os

print(os.listdir("../input"))



import keras



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,multiply, LocallyConnected2D, Lambda)

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

from keras.losses import binary_crossentropy, categorical_crossentropy

from keras.utils import Sequence, to_categorical

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input



import matplotlib.pyplot as plt

import PIL

import cv2

import tensorflow as tf

import pandas as pd

import imgaug as ia

import numpy as np



from imgaug import augmenters as iaa



from PIL import Image, ImageOps

from sklearn.utils import class_weight, shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score



%config InlineBackend.figure_format="svg"

%matplotlib inline
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
import warnings

warnings.filterwarnings("ignore")

SIZE = 300

NUM_CLASSES = 5

WORKERS=2
x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=8)

y.hist()
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

        random_order=True)

class My_Generator(Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size, is_train=True,

                 mix=False, augment=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

        else:

            pass

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
y = to_categorical(y, num_classes=NUM_CLASSES)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,

                                                      stratify=y, random_state=8)




from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 30; batch_size = 16

checkpoint = ModelCheckpoint('../working/model_.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)



train_generator = My_Generator(train_x, train_y, batch_size, is_train=True)

train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)



net = InceptionResNetV2(include_top=False, weights='../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',

                       input_tensor=None, input_shape=(SIZE, SIZE, 3))
x = net.output

x = Flatten()(x)

x = Dropout(.5)(x)

output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

net_final = Model(inputs=net.input, outputs=output_layer)
FREEZE_LAYERS = 2



for layer in net_final.layers[:FREEZE_LAYERS]:

    layer.trainable = False

for layer in net_final.layers[FREEZE_LAYERS:]:

    layer.trainable = True
net_final.compile(optimizer=Adam(lr=1e-5),

loss='categorical_crossentropy', metrics=['accuracy'])
net_final.summary()
net_final.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=20,

    workers=WORKERS, use_multiprocessing=True,

    verbose=1)
predicted = []
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
from tqdm import tqdm
for i, name in tqdm(enumerate(submit['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    X = np.array((image[np.newaxis])/255)

    score_predict=((net_final.predict(X).ravel()*net_final.predict(X[:, ::-1, :, :]).ravel()*net_final.predict(X[:, ::-1, ::-1, :]).ravel()*net_final.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

    label_predict = np.argmax(score_predict)

    predicted.append(str(label_predict))
submit['diagnosis'] = predicted

submit.to_csv('submission.csv', index=False)

submit.head()