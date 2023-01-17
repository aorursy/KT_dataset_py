import json

import math

import os

import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

import tensorflow as tf

from tqdm import tqdm

%matplotlib inline
np.random.seed(2019)

tf.set_random_seed(2019)
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

train_df['diagnosis'].hist()
test_df.head()
def display_samples(df, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 4*rows))



    for i in range(columns*rows):

        image_path = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        

        

        fig.add_subplot(rows, columns, i+1)

        plt.title(image_id)

        plt.imshow(img)

    

    plt.tight_layout()



display_samples(train_df)
def get_pad_width(im, new_shape, is_rgb=True):

    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    if is_rgb:

        pad_width = ((t,b), (l,r), (0, 0))

    else:

        pad_width = ((t,b), (l,r))

    return pad_width









def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img







def circle_crop(img, sigmaX=45):   

    """

    Create circular crop around image centre    

    """    

    

    img = cv2.imread(img)

    img = crop_image_from_gray(img)    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img=cv2.resize(img, (224, 224))

    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    return img 
N = train_df.shape[0]

x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(train_df['id_code'])):

    x_train[i, :, :, :] = circle_crop(

        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png')
N = test_df.shape[0]

x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(test_df['id_code'])):

    x_test[i, :, :, :] = circle_crop(

        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png')

y_train = pd.get_dummies(train_df['diagnosis']).values



y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))

x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=0.15, 

    random_state=2019

)
class MixupGenerator():

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):

        self.X_train = X_train

        self.y_train = y_train

        self.batch_size = batch_size

        self.alpha = alpha

        self.shuffle = shuffle

        self.sample_num = len(X_train)

        self.datagen = datagen



    def __call__(self):

        while True:

            indexes = self.__get_exploration_order()

            itr_num = int(len(indexes) // (self.batch_size * 2))



            for i in range(itr_num):

                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]

                X, y = self.__data_generation(batch_ids)



                yield X, y



    def __get_exploration_order(self):

        indexes = np.arange(self.sample_num)



        if self.shuffle:

            np.random.shuffle(indexes)



        return indexes



    def __data_generation(self, batch_ids):

        _, h, w, c = self.X_train.shape

        l = np.random.beta(self.alpha, self.alpha, self.batch_size)

        X_l = l.reshape(self.batch_size, 1, 1, 1)

        y_l = l.reshape(self.batch_size, 1)



        X1 = self.X_train[batch_ids[:self.batch_size]]

        X2 = self.X_train[batch_ids[self.batch_size:]]

        X = X1 * X_l + X2 * (1 - X_l)



        if self.datagen:

            for i in range(self.batch_size):

                X[i] = self.datagen.random_transform(X[i])

                X[i] = self.datagen.standardize(X[i])



        if isinstance(self.y_train, list):

            y = []



            for y_train_ in self.y_train:

                y1 = y_train_[batch_ids[:self.batch_size]]

                y2 = y_train_[batch_ids[self.batch_size:]]

                y.append(y1 * y_l + y2 * (1 - y_l))

        else:

            y1 = self.y_train[batch_ids[:self.batch_size]]

            y2 = self.y_train[batch_ids[self.batch_size:]]

            y = y1 * y_l + y2 * (1 - y_l)



        return X, y
BATCH_SIZE = 32



def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.15,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

    )



# Using original generator

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)

# Using Mixup

mixup_generator = MixupGenerator(x_train, y_train, batch_size=BATCH_SIZE, alpha=0.2, datagen=create_datagen())()
class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_val = y_val.sum(axis=1) - 1

        

        y_pred = self.model.predict(X_val) > 0.5

        y_pred = y_pred.astype(int).sum(axis=1) - 1



        _val_kappa = cohen_kappa_score(

            y_val,

            y_pred, 

            weights='quadratic'

        )



        self.val_kappas.append(_val_kappa)



        print(f"val_kappa: {_val_kappa:.4f}")

        

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model.save('model.h5')



        return
from keras.applications import DenseNet169

densenet = DenseNet169(

    weights='../input/densenet-keras/DenseNet-BC-169-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)
def build_model():

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.0005),

        metrics=['accuracy']

    )

    

    return model
model = build_model()

model.summary()
kappa_metrics = Metrics()




history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

    epochs=100,

    validation_data=(x_val, y_val),

    callbacks=[kappa_metrics]

)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
model.load_weights('model.h5')

y_val_pred = model.predict(x_val)



def compute_score_inv(threshold):

    y1 = y_val_pred > threshold

    y1 = y1.astype(int).sum(axis=1) - 1

    y2 = y_val.sum(axis=1) - 1

    score = cohen_kappa_score(y1, y2, weights='quadratic')

    

    return 1 - score



simplex = scipy.optimize.minimize(

    compute_score_inv, 0.5, method='nelder-mead'

)



best_threshold = simplex['x'][0]
y_test = model.predict(x_test) > 0.5

y_test = y_test.astype(int).sum(axis=1) - 1



test_df['diagnosis'] = y_test
def xdisplay_samples(df, columns=4, rows=12):

    fig=plt.figure(figsize=(5*columns, 4*rows))



    for i in range(columns*rows):

        image_path = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        #img = cv2.imread(f'../input/aptos2019-blindness-detection/test_images/{image_path}.png')

        image = circle_crop(f'../input/aptos2019-blindness-detection/test_images/{image_path}.png')

        

        

        fig.add_subplot(rows, columns, i+1)

        plt.title(image_id)

        plt.imshow(image)

    

    plt.tight_layout()



xdisplay_samples(test_df)
model.save("modelx.h5")
test_df.to_csv('submission.csv',index=False)