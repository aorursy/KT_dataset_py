### Transfer Learning using DenseNet-121-B5 ####



### Kudos and Thanks to all Kaggle contributors, to make awesome datasets available ####
# just to see the input dirs



!ls ../input/



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

# import sys

# package_path = '../input/kerasradam/repository/CyberZHG-keras-radam-78dcb11/keras_radam'

# sys.path.append(package_path)



# print(sys.path)
# from radam import RAdam
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import json

import math

import os



import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

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

from keras.applications.densenet import DenseNet121

import seaborn as sns

sns.set()





from IPython.display import display





%matplotlib inline



# Keep iterating and changing. Grid search would be a better option here !

EPOCHS = 150

BATCH_SIZE = 16

SEED = 20031976

LRATE = 0.0001

VERBOSE=0



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
np.random.seed(SEED)

tf.set_random_seed(SEED)



train_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv')

print("Datasets loaded..")



print ('Train DF Shape', train_df.shape)
# Data

display(train_df.head(2))

display(test_df.head(2))



# Print Shape of Data

print("train_df shape = ",train_df.shape)

print("test_df shape = ",test_df.shape)



# Distribution of Training Data

display(train_df['diagnosis'].value_counts())

sns.countplot(train_df['diagnosis'], color='black')

def display_image(df, rows, columns):

    fig=plt.figure(figsize=(10, 10))



    for i in range(columns*rows):

        image_path = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = cv2.imread(f'/kaggle/input/aptos2019-blindness-detection/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        fig.add_subplot(rows, columns, i+1)

        plt.title(image_id)

        plt.imshow(img)

    

    plt.tight_layout()

    

def display_single_image(img):

    fig=plt.figure(figsize=(10, 10))

    plt.title('Sample Img')

    plt.imshow(img)



display_image(train_df, 4, 4)



# Now, I'm observing black borders around the retina, which is not of much interest.



print('Train DF Shape:',train_df.shape)
def load_ben_color(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    return image



def preprocess_image(image_path, desired_size=224):

    # Add Lighting, to improve quality

    im = load_ben_color(image_path)

    return im

    



# Trail-1 Under sampling by deleting oversized classes (Class:0)

def under_sample_make_all_same(df, categories, max_per_category):

    df = pd.concat([df[df['diagnosis'] == c][:max_per_category] for c in categories])

    df = df.sample(n=(max_per_category)*len(categories), replace=False, random_state=20031976)

    df.index = np.arange(len(df))

    return df

# train_df = under_sample_make_all_same(train_df,[0,1,2,3,4], 193 ) 

#Under-sample class-0 (1805-805=1000) and Over-sample other classes so each class has 1000 entries

print('Train DF Shape:',train_df.shape)

train_df = train_df.drop(train_df[train_df['diagnosis'] == 0].sample(n=805, replace=False).index)



N = train_df.shape[0]

x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

#tqdm

for i, image_id in enumerate((train_df['id_code'])):

    x_train[i, :, :, :] = preprocess_image(

        f'/kaggle/input/aptos2019-blindness-detection/train_images/{image_id}.png'

    )

#     print('Preprocessing Image:',i)

    

    

N = test_df.shape[0]

x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)



for i, image_id in enumerate((test_df['id_code'])):

    x_test[i, :, :, :] = preprocess_image(

        f'/kaggle/input/aptos2019-blindness-detection/test_images/{image_id}.png'

    )

#     print('Preprocessing Test Image:',i)

    

y_train = pd.get_dummies(train_df['diagnosis']).values

# View the sample pre-processed images here

def display_single_image(img):

    fig=plt.figure(figsize=(10, 10))

    plt.title('Sample Img')

    plt.imshow(img)



# Training Images

print('X Train Shape:', x_train.shape)

display_single_image(x_train[0])

display_single_image(x_train[1])





# Testing Images

print('X Test Shape:', x_test.shape)
# Let's crop the images, to their region of interest

# Credits to https://www.kaggle.com/taindow/pre-processing-train-and-test-images

def crop_image_from_gray(img,tol=7):

    

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = gray_img>tol        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0):

            return img

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def circle_crop(img):   

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    

    return img 



circle_crop_img = circle_crop(x_train[1])



display_single_image(circle_crop_img)



# This has some work, to do. Not continuing any more :)
print("x_train.shape=",x_train.shape)

print("y_train.shape=",y_train.shape)

print("x_test.shape=",x_test.shape)

# Trail-2 Over sampling by increasing undersized classes

from imblearn.over_sampling import SMOTE, ADASYN

x_resampled, y_resampled = SMOTE(random_state=SEED).fit_sample(x_train.reshape(x_train.shape[0], -1), train_df['diagnosis'].ravel())



print("x_resampled.shape=",x_resampled.shape)

print("y_resampled.shape=",y_resampled.shape)



x_train = x_resampled.reshape(x_resampled.shape[0], 224, 224, 3)

y_train = pd.get_dummies(y_resampled).values



print("x_train.shape=",x_train.shape)

print("y_train.shape=",y_train.shape)
y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))
# Split 85-15 training-validation sets

x_sptrain, x_spval, y_sptrain, y_spval = train_test_split(

    x_train, y_train_multi, 

    test_size=0.10, 

    random_state=SEED

)

print("train-validation splitted ...")
def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.10,        # set range for random zoom

        fill_mode='constant',   # set mode for filling points outside the input boundaries

        cval=0.,                # value used for fill_mode = "constant"

        horizontal_flip=True,   # randomly flip images

        vertical_flip=True,     # randomly flip images

#         rotation_range=20       # Degree range for random rotations

    )



# Using original generator

data_generator = create_datagen().flow(x_sptrain, y_sptrain, batch_size=BATCH_SIZE, seed=SEED)

print("Image data augmentated ...")
# Define evaluation metrics



import keras.backend as K



def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def fbeta_score(y_true, y_pred, beta=1):

    if beta < 0:

        raise ValueError('The lowest choosable beta is zero (only precision).')



    # If there are no true positives, fix the F score at 0 like sklearn.

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0



    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    bb = beta ** 2

    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score



def fmeasure(y_true, y_pred):

    return fbeta_score(y_true, y_pred, beta=1)



def mean_pred(y_true, y_pred):

    return K.mean(y_pred)



def f1_score(y_true, y_pred):

    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    return 2*(p*r) / (p+r+K.epsilon())



print("Evaluation metrics defined ...")

from keras.applications import DenseNet169,DenseNet121



# Transfer Learning.. This is so COOL !!!

densenet = DenseNet121(

    weights='/kaggle/input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)



model = Sequential()

model.add(densenet)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dropout(0.2))

model.add(layers.Dense(5, activation='sigmoid'))



model.compile(

    loss='binary_crossentropy',

    optimizer=Adam(lr=LRATE),

    metrics=['accuracy',mean_pred, precision, recall, f1_score, fbeta_score, fmeasure]

)

model.summary()
# callback to keep track of kappa score during training

class KappaMetrics(Callback):

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



        print(f"Epoch: {epoch+1} val_kappa: {_val_kappa:.4f}")

        

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model.save('model.h5')



        return

    

kappa_score = KappaMetrics()







history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(x_spval, y_spval),

    callbacks=[kappa_score],

    verbose=VERBOSE

)    
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df.head(EPOCHS)
y_test = model.predict(x_test) > 0.5

y_test = y_test.astype(int).sum(axis=1) - 1



test_df['diagnosis'] = y_test

test_df.to_csv('submission.csv',index=False)

display(test_df.head(5))


