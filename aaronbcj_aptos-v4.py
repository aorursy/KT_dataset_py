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



EPOCHS = 50

BATCH_SIZE = 16

SEED = 20031976

LRATE = 0.00005

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
# Data

display(train_df.head(2))

display(test_df.head(2))



# Shape of data

print("train_df shape = ",train_df.shape)

print("test_df shape = ",test_df.shape)



# Distribution of data

display(train_df['diagnosis'].value_counts())

sns.countplot(train_df['diagnosis'], color='gray')
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



display_image(train_df, 4, 4)
def preprocess_image(image_path, desired_size=224):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    return im



# Trail-1 Under sampling by deleting oversized classes (Class:0)

def under_sample_make_all_same(df, categories, max_per_category):

    df = pd.concat([df[df['diagnosis'] == c][:max_per_category] for c in categories])

    df = df.sample(n=(max_per_category)*len(categories), replace=False, random_state=20031976)

    df.index = np.arange(len(df))

    return df

#train_df = under_sample_make_all_same(train_df,[0,1,2,3,4], 193 ) 

#Under-sample class-0 (1805-805=1000) and Over-sample other classes so each class has 1000 entries

train_df = train_df.drop(train_df[train_df['diagnosis'] == 0].sample(n=805, replace=False).index)



N = train_df.shape[0]

x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

#tqdm

for i, image_id in enumerate((train_df['id_code'])):

    x_train[i, :, :, :] = preprocess_image(

        f'/kaggle/input/aptos2019-blindness-detection/train_images/{image_id}.png'

    )

    

N = test_df.shape[0]

x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)



for i, image_id in enumerate((test_df['id_code'])):

    x_test[i, :, :, :] = preprocess_image(

        f'/kaggle/input/aptos2019-blindness-detection/test_images/{image_id}.png'

    )

    

y_train = pd.get_dummies(train_df['diagnosis']).values



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



# Trail-3 No sampling



# Each class should have 1000 samples now (5 x 1000 = 5000)

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

        #rotation_range=20       # Degree range for random rotations

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
# Transfer Learning

densenet = DenseNet121(

    weights='/kaggle/input/densenet121/DenseNet-BC-121-32-no-top.h5',

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
#history_df[['loss', 'val_loss']].plot()

#history_df[['acc', 'val_acc']].plot()

#history_df[['acc', 'precision', 'recall', 'f1_score', 'fbeta_score', 'fmeasure']].plot()



f1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))

t1 = f1.suptitle('CNN Performance', fontsize=12)

f1.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,EPOCHS + 1))

ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, EPOCHS + 1, 5))

ax1.set_ylabel('Accuracy %')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, EPOCHS + 1, 5))

ax2.set_ylabel('Loss %')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")



ax3.plot(epoch_list, history.history['acc'], label='Accuracy')

ax3.plot(epoch_list, history.history['precision'], label='Precision')

ax3.plot(epoch_list, history.history['recall'], label='Recall')

ax3.plot(epoch_list, history.history['f1_score'], label='F1 score')

ax3.plot(epoch_list, history.history['fbeta_score'], label='Fbeta score')

ax3.plot(epoch_list, history.history['fmeasure'], label='FMeasure')

ax3.set_xticks(np.arange(0, EPOCHS + 1, 5))

ax3.set_ylabel('Score')

ax3.set_xlabel('Epoch')

ax3.set_title('Performance')

l3 = ax3.legend(loc="best")



ax4.plot(epoch_list, kappa_score.val_kappas, label='Kappa score')

ax4.set_xticks(np.arange(0, EPOCHS + 1, 5))

ax4.set_ylabel('Score')

ax4.set_xlabel('Epoch')

ax4.set_title('Kappa Metrics')

l4 = ax4.legend(loc="best")



display("Maximum Kappa Score: %s" %max(kappa_score.val_kappas))
y_test = model.predict(x_test) > 0.5

y_test = y_test.astype(int).sum(axis=1) - 1



test_df['diagnosis'] = y_test

test_df.to_csv('submission.csv',index=False)

display(test_df.head(5))



import datetime

print("Ran at UTC : ", datetime.datetime.utcnow())
