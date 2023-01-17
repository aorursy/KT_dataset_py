# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import cv2

import matplotlib.pyplot as plt

print(os.listdir("../input"))

from tqdm import tqdm

from tensorflow import keras

from sklearn.pipeline import Pipeline

np.random.seed(123)

# import keras_metrics

# Any results you write to the current directory are saved as output.
from sklearn.base import TransformerMixin



# This is just a quick and simple StandardScaler transformer which works with 3 dimensional inputs like we are giving our model

# read more about StandardScaler here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

class CustomStandardScalerForCnn(TransformerMixin):

    def __init__(self, with_mean=True, with_std=True):

        self.with_mean = with_mean

        self.with_std = with_std

        self.mean_ = None

        self.std_ = None

        

    def fit(self, X, y=None):

        if self.with_mean:

            self.mean_ = X.mean()

        else:

            self.mean_ = 0

            

        if self.with_std:

            self.std_ = X.std()

        else:

            self.std_ = 1

        

        return self

    

    def transform(self, X):

        if self.mean_ and self.std_:

            return (X - self.mean_) / self.std_

        else:

            raise("CustomStandardScalerForCnn is not fitted")

            

    def inverse_transform(self, X):

        if self.with_std:

            X *= self.std_

        if self.with_mean:

            X += self.mean_

        return X
import glob

all_imges = glob.glob(os.path.join('..', 'input', 'chest_xray', 'chest_xray', '*', '*', '*.jpeg'))

len(all_imges)
def load_img(img_path, img_shape=128):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (img_shape, img_shape))

    return img
def load_imges(class_label, dataset_type='train', img_shape=128):

    path_imges = glob.glob(os.path.join('..', 'input', 'chest_xray', 'chest_xray', dataset_type, class_label, '*.jpeg'))

    x_data = np.array([load_img(p, img_shape) for p in path_imges])

    y_labels = [class_label] * len(path_imges)

    return x_data, y_labels
def Load(dataset_type, img_shape=128):

    normal_data, normal_labels = load_imges('NORMAL', dataset_type, img_shape)

    PNEUMONIA_data, PNEUMONIA_labels = load_imges('PNEUMONIA', dataset_type, img_shape)



    data = np.vstack((normal_data, PNEUMONIA_data))

    labels = np.array((normal_labels + PNEUMONIA_labels))

    return data, labels #.reshape(labels.shape[0], 1)

os.listdir(os.path.join('..', 'input', 'chest_xray', 'chest_xray', 'train'))

img = load_img(os.path.join('..', 'input', 'chest_xray', 'chest_xray', 'train', 'NORMAL', 'NORMAL2-IM-1232-0001.jpeg'))

plt.imshow(img);
x_train, y_train = Load(dataset_type='train')

x_val, y_val = Load(dataset_type='val')

x_test, y_test = Load(dataset_type='test')
np.unique(y_train)

from sklearn.utils import class_weight



balanced_class_weight_array = class_weight.compute_class_weight('balanced',

                                                                np.unique(y_train),

                                                                y_train)

balanced_class_weight = dict(zip(range(len(np.unique(y_train))), balanced_class_weight_array))

balanced_class_weight
def keras_build(input_shape):

    clf = keras.models.Sequential([

        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),

        keras.layers.MaxPool2D(pool_size=(2, 2)),

        keras.layers.Dropout(0.5),



        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),

        keras.layers.MaxPool2D(pool_size=(2, 2)),

        keras.layers.Dropout(0.5),



        keras.layers.Flatten(),

        keras.layers.Dense(256, activation='relu'),

        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(1, activation='sigmoid')

        ])



    clf.compile(optimizer=keras.optimizers.Adam(lr=0.0005),

                loss='binary_crossentropy',

                metrics=['accuracy'])



    return clf



keras_build(input_shape=x_train[0].shape).summary()
model = Pipeline([

    ('scaler', CustomStandardScalerForCnn()),

    ('keras', keras.wrappers.scikit_learn.KerasClassifier(keras_build,

                                                          input_shape=x_train[0].shape,

                                                          shuffle=True,

                                                          epochs=100,

                                                          class_weight=balanced_class_weight,

                                                          batch_size=128,

                                                          validation_split=0.1,

                                                          callbacks=[

                                                              keras.callbacks.EarlyStopping(patience=5),

                                                              keras.callbacks.ReduceLROnPlateau(monitor='val_loss',

                                                                                                factor=0.5, patience=3,

                                                                                                verbose=1, min_lr=0)

                                                                    ],             

                                                          verbose=1))

])





model.fit(x_train, y_train)


from sklearn import metrics

y_train_pred = model.predict(x_train)



print(metrics.classification_report(y_true=y_train, y_pred=y_train_pred))

print(metrics.accuracy_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = model.predict(x_test)



print(metrics.classification_report(y_true=y_test, y_pred=y_test_pred))

print(metrics.accuracy_score(y_true=y_test, y_pred=y_test_pred))
idx = 622

print(y_test[idx])

plt.imshow(x_test[idx])
import lime

from lime import lime_image

from skimage.segmentation import mark_boundaries
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(x_test[idx], model.predict_proba, top_labels=2, hide_color=0, num_samples=1000)

source_image, mask = explanation.get_image_and_mask(1, positive_only=False)
plt.imshow(mark_boundaries(source_image, mask))
conv_base = keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=x_train[0].shape)

# conv_base = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=x_train[0].shape)

conv_base.trainable = False
def transfer_build():

    clf = keras.models.Sequential([

        conv_base,

        keras.layers.Flatten(),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(256, activation='relu'),

        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(1, activation='sigmoid')

        ])



    clf.compile(optimizer=keras.optimizers.Adam(lr=0.0005),

                loss='binary_crossentropy',

                metrics=['accuracy'])



    return clf



transfer_build().summary()
model_transfer = Pipeline([

    ('scaler', CustomStandardScalerForCnn()),

    ('keras', keras.wrappers.scikit_learn.KerasClassifier(transfer_build,

                                                          shuffle=True,

                                                          epochs=100,

                                                          class_weight=balanced_class_weight,

                                                          batch_size=128,

                                                          validation_split=0.1,

                                                          callbacks=[

                                                              keras.callbacks.EarlyStopping(patience=5),

                                                              keras.callbacks.ReduceLROnPlateau(monitor='val_loss',

                                                                                                factor=0.5, patience=3,

                                                                                                verbose=1, min_lr=0)

                                                                    ],             

                                                          verbose=1))

])





model_transfer.fit(x_train, y_train)
y_train_pred = model_transfer.predict(x_train)



print(metrics.classification_report(y_true=y_train, y_pred=y_train_pred))

print(metrics.accuracy_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = model_transfer.predict(x_test)



print(metrics.classification_report(y_true=y_test, y_pred=y_test_pred))

print(metrics.accuracy_score(y_true=y_test, y_pred=y_test_pred))