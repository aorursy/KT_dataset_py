import os

import zipfile

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

import tensorflow as tf

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import Adam

from keras.applications import MobileNet

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

pd.set_option('display.max_columns', 99)
img_width, img_height = 128, 128
# <SPECIFY YOUR PATH HERE>

path = "../input/kaggledays-china/"

os.listdir(path) # you should have test and train data here



# This initial size of the pictures - change it, if you're going to crop them manually





# Specify here where your data are located - I use combined, 3-channel data



train_data_dir = path + 'train3c/train3c' # train data - it should have directories for each class inside

test_data_dir = path + 'test3c'  # test data - you have to keep 2-level directory structure here



nb_train_samples = 5024

nb_validation_samples = 1257

epochs = 60

batch_size = 512

nb_test_samples = len(pd.read_csv(path + 'test.csv'))
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)



# I use sklearn's implementation of ROC AUC, out of convenience

def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

#     shear_range=0.2,

    zoom_range=[0.8,1.0],

    brightness_range=[0.8,1.0],

    rotation_range=90,

    horizontal_flip=True,

    vertical_flip=True,

    validation_split=0.15

)

train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary',

    seed = 42,

    subset='training'

)

valid_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary',

    seed = 42,

    subset='validation'

)

X, y = next(train_generator)
# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(

    rescale=1. / 255,

    zoom_range=[0.8,1.0],

    brightness_range=[0.8,1.0],

    rotation_range=90,

    horizontal_flip=True,

    vertical_flip=True

)



# avoid shuffling her - it'll make hard keeping your predictions and labels together

test_generator = test_datagen.flow_from_directory(

    test_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode=None,

    seed = 42,

    shuffle = False

)
X, y = next(train_generator)

X.shape, y.shape

stars = X[y == 1]

nonstars = X[y == 0]

stars.shape, nonstars.shape
k = 5

fig, axs = plt.subplots(nrows=k, ncols=k, sharex=True, sharey=True)

for i in  range(k * k):

    axs[i //k, i % k].imshow(stars[i])

plt.tight_layout()

plt.suptitle('Stars')

plt.show();
k = 5

fig, axs = plt.subplots(nrows=k, ncols=k, sharex=True, sharey=True)

for i in  range(k * k):

    axs[i //k, i % k].imshow(nonstars[i])

plt.tight_layout()

plt.suptitle('Non Stars')

plt.show();
def get_model():

    K.clear_session()

    base_model = MobileNet(weights='imagenet', input_shape=(128, 128, 3),

                  include_top=False, pooling='avg')

    x = base_model.output

    y_pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=y_pred)



model = get_model()

print(len(model.layers))

optimizer = Adam(lr=0.0003)

model.compile(loss='binary_crossentropy',

              optimizer=optimizer,

              metrics=[auroc])

model.summary()
hists = []

hist = model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    validation_data=valid_generator,

    epochs=epochs,

)

hists.append(hist)
hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)

hist_df.index = np.arange(1, len(hist_df)+1)

fig, ax = plt.subplots(figsize=(16, 10))

ax.plot(hist_df.val_auroc, lw=5, label='Validation Accuracy')

ax.plot(hist_df.auroc, lw=5, label='Training Accuracy')

ax.set_ylabel('AUC')

ax.set_xlabel('Epoch')

ax.grid()

plt.legend(loc=0)

fig.savefig('hist.png', dpi=300)

plt.show();
pred = np.zeros(nb_test_samples)

n_tta = 30

for _ in range(n_tta):

    p = model.predict_generator(test_generator, verbose=1) 

    pred += np.array([x[0] for x in p]) / n_tta

pred[:10] # check if format is correct





labels = (test_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())



# Save the data to required sumbission format



filenames=test_generator.filenames

results=pd.DataFrame({"id":[x.split("/")[2].split(".")[0] for x in filenames],

                      "is_star":pred})

results.to_csv("results.csv", index=False)
results.head()