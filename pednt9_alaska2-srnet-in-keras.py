import pandas as pd

import numpy as np

import os

import gc

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from tqdm.notebook import tqdm

import random

import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

from time import time

from collections import Counter



from PIL import Image

from random import shuffle



import tensorflow as tf

from tensorflow.keras.metrics import AUC



import keras

import keras.backend as K

import keras.layers as L

from keras import Model

from keras.utils import plot_model

from keras.preprocessing.image import ImageDataGenerator



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
PATH = "/kaggle/input/alaska2-image-steganalysis/"

IMAGE_IDS = os.listdir(os.path.join(PATH, 'Cover'))

N_IMAGES = len(IMAGE_IDS)

ALGORITHMS = ['JMiPOD', 'JUNIWARD', 'UERD']

IMG_SIZE = 256



sample_sub = pd.read_csv(PATH + 'sample_submission.csv')
# To make things reproductible



def seed_everything(seed=42):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    os.environ['TF_KERAS'] = '1'



seed_everything()
def layer_type1(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):

    x = L.Conv2D(filters, kernel_size, padding="same")(x_inp)

    x = L.BatchNormalization()(x)

    x = L.ReLU()(x)

    if dropout_rate > 0:

        x = L.Dropout(dropout_rate)(x)



    return x



def layer_type2(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):

    x = layer_type1(x_inp, filters)

    x = L.Conv2D(filters, kernel_size, padding="same")(x)

    x = L.BatchNormalization()(x)

    x = L.ReLU()(x)

    if dropout_rate > 0:

        x = L.Dropout(dropout_rate)(x)



    x = L.Add()([x, x_inp])

    

    return x



def layer_type3(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):

    x = layer_type1(x_inp, filters)

    x = L.Conv2D(filters, kernel_size, padding="same")(x)

    x = L.BatchNormalization()(x)

    x = L.ReLU()(x)

    x = L.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    if dropout_rate > 0:

        x = L.Dropout(dropout_rate)(x)

        

    x_res = L.Conv2D(filters, kernel_size, strides=(2, 2))(x_inp)

    x_res = L.BatchNormalization()(x_res)

    if dropout_rate > 0:

        x_res = L.Dropout(dropout_rate)(x_res)



    x = L.Add()([x, x_res])

    

    return x



def layer_type4(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):

    x = layer_type1(x_inp, filters)

    x = L.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)

    x = L.BatchNormalization()(x)

    if dropout_rate > 0:

        x = L.Dropout(dropout_rate)(x)    

    x = L.GlobalAveragePooling2D()(x)

    

    return x
def alaska_weighted_auc(y_true, y_valid):

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights =        [       2,   1]

    

    fpr, tpr, thresholds = roc_curve(y_true, y_valid)

    

    # size of subsets

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

    normalization = np.dot(areas, weights)

    

    try:

    

        competition_metric = 0

        for idx, weight in enumerate(weights):

            y_min = tpr_thresholds[idx]

            y_max = tpr_thresholds[idx + 1]

            mask = (y_min < tpr) & (tpr < y_max)



            x_padding = np.linspace(fpr[mask][-1], 1, 100)



            x = np.concatenate([fpr[mask], x_padding])

            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

            y = y - y_min # normalize such that curve starts at y=0

            score = auc(x, y)

            submetric = score * weight

            best_subscore = (y_max - y_min) * weight

            competition_metric += submetric

    except:

        # sometimes there's a weird bug so return naive score

        return .5

        

    return competition_metric / normalization



def alaska_tf(y_true, y_val):

    """Wrapper for the above function"""

    return tf.py_function(func=alaska_weighted_auc, inp=[y_true, y_val], Tout=tf.float32)
def make_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_type2=5, dropout_rate=0):

    # I reduced the size (image size, filters and depth) of the original network because it was way to big

    inputs = L.Input(shape=input_shape)

    

    x = layer_type1(inputs, filters=64, dropout_rate=dropout_rate)

    x = layer_type1(x, filters=16, dropout_rate=dropout_rate)    

    

    for _ in range(num_type2):

        x = layer_type2(x, filters=16, dropout_rate=dropout_rate)         

    

    x = layer_type3(x, filters=16, dropout_rate=dropout_rate) 

    x = layer_type3(x, filters=32, dropout_rate=dropout_rate)            

    x = layer_type3(x, filters=64, dropout_rate=dropout_rate)            

    #x = layer_type3(x, filters=128, dropout_rate=dropout_rate) 

    

    x = layer_type4(x, filters=128, dropout_rate=dropout_rate)        

    

    x = L.Dense(64)(x)

    x = L.BatchNormalization()(x)

    x = L.ReLU()(x)

    if dropout_rate > 0:

        x = L.Dropout(dropout_rate)(x)



    predictions = L.Dense(1, activation="sigmoid")(x)

    

    model = Model(inputs=inputs, outputs=predictions)

    keras_auc = AUC()

    

    model.compile(optimizer='adam',

                  loss='binary_crossentropy', 

                  metrics=[alaska_tf])

    

    return model

    
model = make_model(num_type2=4, dropout_rate=0.1)

model.summary()
plot_model(model, show_shapes=True, to_file="model.png")
def load_image(data):

    i, j, img_path, labels = data

    

    img = Image.open(img_path)

    img = img.convert('RGB')

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

    

    label = labels[i][j]

    

    return [np.array(img), label]



def load_training_data_multi(n_images=100):

    train_data = []

    data_paths = [os.listdir(os.path.join(PATH, alg)) for alg in ['Cover'] + ALGORITHMS]

    labels = [np.zeros(N_IMAGES), np.ones(N_IMAGES), np.ones(N_IMAGES), np.ones(N_IMAGES)]

    

    print('Loading...')

    for i, image_path in enumerate(data_paths):

        print(f'\t {i+1}-th folder')

        

        train_data_alg = joblib.Parallel(n_jobs=4, backend='threading')(

            joblib.delayed(load_image)([i, j, os.path.join(PATH, [['Cover'] + ALGORITHMS][0][i], img_p), labels]) for j, img_p in enumerate(image_path[:n_images]))



        train_data.extend(train_data_alg)

        

    shuffle(train_data)

    return train_data
def load_test_data():

    test_data = []

    for img_p in os.listdir(os.path.join(PATH, 'Test')):

        img = Image.open(os.path.join(PATH, 'Test', img_p))

        img = img.convert('RGB')

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

        test_data.append([np.array(img)])

            

    return test_data
start = time()

training_data = load_training_data_multi(n_images=5000)



trainImages = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

trainLabels = np.array([i[1] for i in training_data], dtype=int)



X_train, X_val, y_train, y_val = train_test_split(trainImages, trainLabels, random_state=42, stratify=trainLabels)



# Then save some RAM

del training_data

del trainImages

del trainLabels

gc.collect()



print(f"{(time() - start) / 60: .2f} min elapsed.")
model.fit(X_train, y_train,

          validation_data=(X_val, y_val),

          batch_size=16,

          epochs=5, 

          verbose=1)
test = load_test_data()
test_images = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
predict = model.predict(test_images, batch_size=64)
sample_sub['Label'] = predict
sample_sub.head()
sample_sub.to_csv('submission.csv', index=False)