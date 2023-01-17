import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.pipeline import Pipeline

import os

import random

import matplotlib.pyplot as plt

from tensorflow import keras

np.random.seed(123)

%matplotlib inline
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

    

        
def VGG_inspired_build():

    clf = keras.models.Sequential([

        keras.layers.ZeroPadding2D((1,1), input_shape=(32, 32, 3)),

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

        keras.layers.ZeroPadding2D((1,1)),

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

        keras.layers.MaxPooling2D(pool_size=(2, 2)), # stride=2

        keras.layers.ZeroPadding2D((1,1)),

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

        keras.layers.ZeroPadding2D((1,1)),

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

        keras.layers.MaxPooling2D(pool_size=(2, 2)), # stride=2

        keras.layers.ZeroPadding2D((1,1)),

        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),

        keras.layers.ZeroPadding2D((1,1)),

        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),

        keras.layers.MaxPooling2D(pool_size=(2, 2)), # stride=2

        keras.layers.Flatten(),

        keras.layers.Dense(2048, activation='relu'),

        keras.layers.Dropout(0.75),

        keras.layers.Dense(2048, activation='relu'),

        keras.layers.Dropout(0.75),

        keras.layers.Dense(33, activation='softmax')

    ])

    

    

    clf.compile(optimizer=keras.optimizers.Adam(),

                loss='categorical_crossentropy',

                metrics=['accuracy'])



    return clf
def plot_data(img_array, labels_array, n_samples=5):

    classes = set(labels_array)

    fig, ax = plt.subplots(len(classes), n_samples)

    

    for label_idx, class_label in enumerate(classes):

        data = img_array[labels_array == class_label]

        samples_from_class = random.choices(data, k=n_samples)

        for i, img in enumerate(samples_from_class):

            ax[label_idx, i].imshow(img)

            ax[label_idx, i].axis('off')
letters1 = pd.read_csv(os.path.join('..', 'input', 'letters.csv'))

letters1['source_folder'] = 'letters'

letters2 = pd.read_csv(os.path.join('..', 'input', 'letters2.csv'))

letters2['source_folder'] = 'letters2'

letters3 = pd.read_csv(os.path.join('..', 'input', 'letters3.csv'))

letters3['source_folder'] = 'letters3'

letters = pd.concat([letters1, letters2, letters3], ignore_index=True)



letters.head()
%%time

# This is not the most efficient way to load the data but it's ok for now (should only take 30 sec)

import cv2

X = []

y = []

for i, row in letters.iterrows():

    source_folder = row['source_folder']

    img_name = row['file']

    img_arr = cv2.imread(os.path.join('..', 'input', source_folder, img_name))

    if img_arr.shape == (32, 32, 3): # There are some photos in a weird size, I will not use them.

        X.append(img_arr)

        y.append(row['letter'])



X = np.array(X)

y = np.array(y)

X.shape, y.shape
plot_data(X, y, n_samples=25)

plt.subplots_adjust(right=3, top=3)
np.random.seed(123)

model = Pipeline([

    ('scaler', CustomStandardScalerForCnn()),

    ('keras', keras.wrappers.scikit_learn.KerasClassifier(VGG_inspired_build,

                                                          epochs=40,

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



print(VGG_inspired_build().summary())

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)



model.fit(x_train, y_train)
print('score on train data:', model.score(x_train, y_train))

print('score on test data:', model.score(x_test, y_test))

y_pred = model.predict(x_test)

print(metrics.classification_report(y_test, y_pred))