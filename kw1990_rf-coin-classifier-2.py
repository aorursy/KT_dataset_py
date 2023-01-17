from glob import glob

import os

import pandas as pd

from skimage.io import imread

from scipy.ndimage import zoom

import matplotlib.pyplot as plt

    

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

import numpy as np

def imread_size(in_path):

    t_img = imread(in_path)

    return zoom(t_img, [96/t_img.shape[0], 96/t_img.shape[1]]+([1] if len(t_img.shape)==3 else []),

               order = 2)
base_img_dir = os.path.join('..', 'input')

all_training_images = glob(os.path.join(base_img_dir, '*', '*.png'))

full_df = pd.DataFrame(dict(path = all_training_images))

full_df['category'] = full_df['path'].map(lambda x: os.path.basename(os.path.dirname(x)))

full_df = full_df.query('category != "valid"')

cat_enc = LabelEncoder()

full_df['category_id'] = cat_enc.fit_transform(full_df['category'])

y_labels = to_categorical(np.stack(full_df['category_id'].values,0))

print(y_labels.shape)

full_df['image'] = full_df['path'].map(imread_size)

full_df.sample(3)

full_df['category'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(np.stack(full_df['image'].values,0),-1), 

                                                    y_labels,

                                   random_state = 12345,

                                   train_size = 0.75,

                                   stratify = full_df['category'])

print('Training Size', X_train.shape)

print('Test Size', X_test.shape)
from keras.preprocessing.image import ImageDataGenerator # (docu: https://keras.io/preprocessing/image/)



train_datagen = ImageDataGenerator(

        samplewise_std_normalization = True,

        zoom_range=0.2,

        rotation_range = 360,

        )



test_datagen = ImageDataGenerator(

        samplewise_std_normalization = True)



train_gen = train_datagen.flow(X_train, y_train, batch_size=32)



test_gen = train_datagen.flow(X_test, y_test, batch_size=200)
fig, (ax1, ax2) = plt.subplots(2, 4, figsize = (12, 6))

for c_ax1, c_ax2, (train_img, _), (test_img, _) in zip(ax1, ax2, train_gen, test_gen):

    c_ax1.imshow(train_img[0,:,:,0])

    c_ax1.set_title('Train Image')

    

    c_ax2.imshow(test_img[0,:,:,0])

    c_ax2.set_title('Test Image')
y_train_cat = np.argmax(y_train,-1)

y_test_cat = np.argmax(y_test,-1)

print('In Data',X_train.shape,'=>', y_train.shape, '=>', y_train_cat.shape)

print('In Data',X_test.shape,'=>', y_test.shape, '=>', y_test_cat.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import FunctionTransformer, Normalizer

from sklearn.pipeline import make_pipeline

full_pipeline = make_pipeline(

    FunctionTransformer(lambda x: x.reshape((x.shape[0],-1)), validate = False), 

    Normalizer(),

    RandomForestClassifier()

)
%%time

full_pipeline.fit(X_train, y_train_cat)
%%time

y_train_pred = full_pipeline.predict(X_train)

y_pred = full_pipeline.predict(X_test)
from sklearn.metrics import classification_report, classification

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

ax1.matshow(classification.confusion_matrix(y_train_cat, y_train_pred))

ax1.set_title('Training Results')

ax2.matshow(classification.confusion_matrix(y_test_cat, y_pred))

ax2.set_title('Validation Results')
print(classification_report(y_test_cat, y_pred))
from sklearn.metrics import accuracy_score



accuracy_score(y_test_cat, y_pred)