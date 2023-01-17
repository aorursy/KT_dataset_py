# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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

from sklearn.metrics import cohen_kappa_score, accuracy_score, auc, roc_auc_score, roc_curve

import sklearn

import scipy

import tensorflow as tf

from tqdm import tqdm

from keras.preprocessing import image

from keras.models import Model

from keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense



%matplotlib inline
!pip install git+https://github.com/qubvel/efficientnet
from efficientnet.tfkeras import EfficientNetB7 as effnetb7
np.random.seed(2019)

tf.random.set_seed(2019)

TEST_SIZE = 0.25

SEED = 2019

BATCH_SIZE = 8
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

train_df.head(7)
x_train = np.load('../input/four-fold-aptos/train_all_four.npy')

x_test = np.load('../input/four-fold-aptos/test_all_four.npy')
y_train = train_df['diagnosis'].values

y_train
y_train_one_hot = pd.get_dummies(train_df['diagnosis']).values



y_train_multi = np.empty(y_train_one_hot.shape, dtype=y_train_one_hot.dtype)

y_train_multi[:, 4] = y_train_one_hot[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train_one_hot[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train_one_hot.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))
y_train_one_hot
y_train_multi
x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=TEST_SIZE, 

    random_state=SEED

)
# y_train_no_multi, y_val_no_multi = train_test_split(

#     y_train, 

#     test_size=TEST_SIZE, 

#     random_state=SEED

# )
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

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=SEED)




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

            self.model.save('effnetb7_model.h5')



        return



x_train[1].shape
base_model = effnetb7(include_top=False,

                     weights = None,

                     input_shape=(224,224,3))

# base_model.load_weights("../input/efficientnet-weights-for-keras/autoaugment/notop/efficientnet-b7_autoaugment_notop.h5")

# base_model.load_weights("../input/efficientnet-weights-for-keras/noisy-student/notop/efficientnet-b7_noisy-student_notop.h5")
# x = base_model.output

# conv = tf.keras.layers.Conv2D(2, 3)(x)

# batch_normal = BatchNormalization()(conv)

# global_avg_pooling = MaxPooling2D()(batch_normal)

# drop_out = Dropout(0.2)(global_avg_pooling)

# dense1 = Dense(1024, activation='relu')(drop_out)

# dense2 = Dense(5, activation = 'sigmoid')(dense1)

# model = Model(inputs = base_model.input, outputs = dense2)







x = base_model.output

batch_normal = BatchNormalization()(x)

global_avg_pooling = GlobalAveragePooling2D()(batch_normal)

drop_out = Dropout(0.2)(global_avg_pooling)

dense1 = Dense(1024, activation='relu')(drop_out)

dense2 = Dense(5, activation = 'sigmoid')(dense1)

model = Model(inputs = base_model.input, outputs = dense2)

# x = base_model.output

# batch_normal = BatchNormalization()(x)

# global_avg_pooling = MaxPooling2D()(batch_normal)

# drop_out = Dropout(0.2)(global_avg_pooling)

# dense1 = Dense(1024, activation='relu')(drop_out)



# # # global_avg_pooling1 = MaxPooling2D()(dense1)

# # drop_out1 = Dropout(0.2)(dense1)

# # dense2 = Dense(1024, activation='relu')(drop_out1)



# # # global_avg_pooling2 = MaxPooling2D()(dense2)

# # drop_out2 = Dropout(0.2)(dense2)

# # dense3 = Dense(1024, activation='relu')(drop_out2)



# dense4 = Dense(5, activation = 'softmax')(dense1)

# model = Model(inputs = base_model.input, outputs = dense4)
for layer in model.layers:

    layer.trainable = True

    

model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')

# Reducing the Learning Rate if result is not improving. 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',

                              verbose=1)



kappa_metrics = Metrics()
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005), metrics=['accuracy','AUC'])
# model.load_weights('../input/effnet-preprocessed/weights/efficientnet-b7_noisy_student_notop_aptos.h5')
history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

    epochs=60,

    validation_data=(x_val, y_val),

    callbacks=[early_stop, reduce_lr]

)
model.save('./efficientnet-b7_one_hot_four_fold_preprocess_aptos.h5')
history.history.keys()
with open('./efficientnet-b7_one_hot_four_fold_preprocess_aptos.json', 'w') as fp:

    json.dump(str(history.history), fp)
# base_model = effnetb7(include_top=False,

#                      weights = None,

#                      input_shape=(224,224,3))





# x = base_model.output

# batch_normal = BatchNormalization()(x)

# global_avg_pooling = GlobalAveragePooling2D()(batch_normal)

# drop_out = Dropout(0.5)(global_avg_pooling)

# dense1 = Dense(1024, activation='relu')(drop_out)

# dense2 = Dense(5, activation = 'sigmoid')(dense1)

# model = Model(inputs = base_model.input, outputs = dense2)



# model.load_weights('../input/effnet-preprocessed/weights/efficientnet-b7_noisy_student_notop_aptos.h5')



# model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005), metrics=['accuracy','AUC'])
y_predict = model.predict(x_val)

y_predict
val_y = y_predict > 0.5

val_y = val_y.astype(int).sum(axis=1) - 1

val_y
y_real = [4 if (list(i)[4]==1) else list(i).index(0)-1 for i in y_val]

# y_real
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

  

actual = y_real

predicted = val_y

results = confusion_matrix(actual, predicted) 

  

print ('Confusion Matrix :')

print(results)

print ('Accuracy Score :',accuracy_score(actual, predicted) )

print ('Report : ')

print (classification_report(actual, predicted))
predict_probab = [y_predict[i][val_y[i]] for i in range(len(val_y))]

# predict_probab
y_val_one_hot = []

for i in range(len(y_real)):

    y_val_one_hot.append(list(np.zeros(5, dtype = 'uint8')))

y_val_one_hot = np.array(y_val_one_hot)

for i in range(y_val_one_hot.shape[0]):

    y_val_one_hot[i][y_real[i]] = 1
def plot_roc(label):

    y_probab = y_predict[:, label]

    y_label = y_val_one_hot[:,label]

    fpr, tpr, thresholds = roc_curve(y_label, y_probab)

    auc = sklearn.metrics.auc(fpr, tpr)

    plt.plot([0,1],[0,1], 'k--')

    plt.plot(fpr,tpr, label = 'AUC SCORE : {:.3f}'.format(auc))

    plt.title('AUC ROC Curve of class '+str(label))

    plt.xlabel('False Positive rate')

    plt.ylabel('True Positive rate')

    plt.legend(loc = 'best')
plot_roc(0)
plot_roc(1)
plot_roc(2)
plot_roc(3)
plot_roc(4)
model.evaluate(x_val,y_val)
y_test_p = model.predict(x_test)
test_y = y_test_p > 0.5

test_y = test_y.astype(int).sum(axis=1) - 1

test_y
cohen_kappa_score(

            y_real,

            val_y, 

            weights='quadratic'

        )
hist = history.history
# hist = history.history
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(hist["loss"], label="loss")

plt.plot(hist["val_loss"], label="val_loss")

# plt.plot(np.argmin(hist["val_loss"]), np.min(hist["val_loss"]), marker="x", color="r",

#          label="best model")

plt.xlabel("Epochs")

plt.ylabel("log_loss")

plt.legend();
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(hist["accuracy"], label="accuracy")

plt.plot(hist["val_accuracy"], label="val_acc")

# plt.plot(np.argmin(hist["val_loss"]), np.min(hist["val_loss"]), marker="x", color="r",

#          label="best model")

plt.xlabel("Epochs")

plt.ylabel("accuracy")

plt.legend();
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(hist["auc"], label="loss")

plt.plot(hist["val_auc"], label="val_loss")

# plt.plot(np.argmin(hist["val_loss"]), np.min(hist["val_loss"]), marker="x", color="r",

#          label="best model")

plt.xlabel("Epochs")

plt.ylabel("auc")

plt.legend();
model.load_weights('../input/effnet-preprocessed/efficientnet-b7_one_hot_four_fold_preprocess_aptos.h5')
model.predict(x_val)
x_val[0].shape