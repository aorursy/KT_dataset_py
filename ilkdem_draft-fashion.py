import numpy as np
import pandas as pd 
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from keras.layers import Conv2D,Dropout,Dense,Flatten,Input,Concatenate, Activation, Convolution2D, MaxPool2D, BatchNormalization,ZeroPadding2D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
NUM_CLASSES = 4
NUM_COLORS = 3
BATCH_SIZE = 64
IMAGE_SIZE = 224
EPOCHS = 100
dfannos = pd.read_csv("../input/deepfashion/image_annos.csv")
dfannos = dfannos[dfannos['category_name'] == 'shorts'].iloc[:,[3,6]]
dfannos.drop_duplicates(inplace=True)
dfannos['image'] = dfannos['image'].apply(lambda x:x[6:])
boundaries = dfannos['bounding_box'].str.split(',',expand=True)
boundaries[0] = boundaries[0].str.slice(start=1)
boundaries[3] = boundaries[3].str.slice(stop=-1)
dfannos[['x','y','w','h']] = boundaries
dfannos[['x','y','w','h']] = dfannos[['x','y','w','h']].astype(int)

print("data size : ",dfannos.shape)
print("data columns : ",dfannos.columns)

X_data = []
y_data = []
for i, annot in tqdm(dfannos.iterrows()):
    if os.path.exists("../input/deepfashion/shorts/shorts/" + annot['image']):
        img = cv2.imread('../input/deepfashion/shorts/shorts/' + annot['image'])
        dfannos.loc[i,'height'] = img.shape[0]
        dfannos.loc[i,'width'] = img.shape[1]
        x = int(annot['x'] / img.shape[1] * IMAGE_SIZE)
        y = int(annot['y'] / img.shape[0] * IMAGE_SIZE)
        w = int(annot['w'] / img.shape[1] * IMAGE_SIZE)
        h = int(annot['h'] / img.shape[0] * IMAGE_SIZE)
        img = cv2.resize(img,dsize=(IMAGE_SIZE,IMAGE_SIZE))
        X_data.append(img)
        y_data.append([x, y, w, h])
X_data = np.array(X_data,dtype = np.uint8)
X_data = X_data.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
y_data = np.array(y_data,dtype = np.uint8)
dfannos = dfannos[dfannos['width'].isnull() == False]
print(dfannos.shape,X_data.shape,y_data.shape)
def draw_image(image_array, bbox):
    if bbox is not None:
        cv2.rectangle(image_array,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
    plt.imshow(image_array)
draw_image(X_data[5],y_data[5])
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images_paths, labels, batch_size=BATCH_SIZE, image_dimensions = (IMAGE_SIZE ,IMAGE_SIZE ,NUM_COLORS),
                 shuffle=False, augment=False):
        self.labels       = labels              # array of labels
        self.images_paths = images_paths        # array of image paths
        self.dim          = image_dimensions    # image dimensions
        self.batch_size   = batch_size          # batch size
        self.shuffle      = shuffle             # shuffle bool
        self.augment      = augment             # augment data bool
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # select data and load images
        labels = np.array([self.labels[k] for k in indexes])
        #images = self.images_paths[indexes]
        images = np.array([self.images_paths[k] for k in indexes])

        #images = [cv2.imread(self.images_paths[k]) for k in indexes]
        
        # preprocess and augment data
        if self.augment == True:
              images = self.augmentor(images)
		
        #images = np.array([preprocess_input(img) for img in images])
        return images, labels
	
	
    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
                [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                #iaa.Flipud(0.2),  # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": (0.7, 1.2), "y": (0.7, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -45 to +45 degrees
                    shear=(-10, 10),  # shear by -16 to +16 degrees
                    order=[0, 1],
                    # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
                                                             n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                       iaa.GaussianBlur((0, 1.0)),
                                       # blur images with a sigma between 0 and 3.0
                                       iaa.AverageBlur(k=(3, 5)),
                                       # blur image using local means with kernel sizes between 2 and 7
                                       iaa.MedianBlur(k=(3, 5)),
                                       # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                               # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                               # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                                       iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                       iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                                              direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0,
                                                         scale=(0.0, 0.01 * 255),
                                                         per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                       iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                       # randomly remove up to 10% of the pixels
                                       iaa.CoarseDropout((0.01, 0.03),
                                                         size_percent=(0.01, 0.02),
                                                         per_channel=0.2),
                               ]),
                               iaa.Invert(0.01, per_channel=True),
                               # invert color channels
                               iaa.Add((-2, 2), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               #iaa.AddToHueAndSaturation((-1, 1)),
                               # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                       iaa.Multiply((0.9, 1.1), per_channel=0.5),
                                       iaa.BlendAlphaFrequencyNoise(
                                               exponent=(-1, 0),
                                               foreground=iaa.Multiply((0.9, 1.1),
                                                                  per_channel=True),
                                               background=iaa.LinearContrast(
                                                       (0.9, 1.1))
                                       )
                               ]),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                                   sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
                ],
                random_order=True
        )
        return seq.augment_images(images)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

train_data = DataGenerator(X_train, y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True)
test_data = DataGenerator(X_test, y_test, batch_size=BATCH_SIZE, augment=True, shuffle=True)

def build_model1():
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', use_bias=False, input_shape=(IMAGE_SIZE,IMAGE_SIZE,NUM_COLORS)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())


    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(NUM_CLASSES))
    return model

def build_model2():
    kwargs     = {'activation':'relu', 'padding':'same'}
    conv_drop  = 0.2
    dense_drop = 0.5
    inp        = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_COLORS))
    
    x = inp
    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPool2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    h = Dropout(dense_drop)(h)
    h = Dense(16, activation='relu')(h)

    v = MaxPool2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    v = Dropout(dense_drop)(v)
    v = Dense(16, activation='relu')(v)

    x = Concatenate()([h,v])
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='linear')(x)

    return Model(inp,x)

model = build_model1()
model.summary()
callbacks = [
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1,min_lr=1e-8, min_delta=0.01, factor=0.5),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.001),
    tf.keras.callbacks.ModelCheckpoint('model2.h5', save_best_only=True, save_weights_only=True)
]

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mae'])

history = model.fit_generator(generator=train_data,
                                            epochs=EPOCHS,
                                            steps_per_epoch=len(train_data),
                                            callbacks=callbacks,
                                            verbose=2,
                                            validation_data=test_data,
                                            validation_steps =len(test_data)
                                           )
y_pred = model.predict(X_test)
image_index = 22

img = X_test[image_index]
bbox = np.array(y_test[image_index]).astype('int')
pred = np.array(y_pred[image_index]).astype('int')
cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
cv2.rectangle(img,(pred[0],pred[1]),(pred[2],pred[3]),(0,0,255),2)
plt.imshow(img)
plt.show()
y_pred[0]
img = plt.imread("../input/deepfashion/test_data/l_20191-9s9889z8-e5x_a1.jpg")
img = cv2.resize(img,dsize=(IMAGE_SIZE,IMAGE_SIZE))
img = img.reshape(IMAGE_SIZE,IMAGE_SIZE,NUM_COLORS)
X_pred = np.expand_dims(img, axis=0)
y_pred = model.predict(X_pred)
pred = np.array(y_pred[0]).astype('int')
cv2.rectangle(img,(pred[0],pred[1]),(pred[2],pred[3]),(0,0,255),2)
plt.imshow(img)
plt.show()