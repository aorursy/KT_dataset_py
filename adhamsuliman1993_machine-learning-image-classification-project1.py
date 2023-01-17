import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg19 import VGG19

from keras.models import Model, Sequential

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization

from sklearn.model_selection import train_test_split

import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import cv2
cancer_labs = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv', dtype=str)



def append_ext(fn):

    return fn+".tif"



cancer_labs["id"]=cancer_labs["id"].apply(append_ext)



print("Cancer image data set count:", cancer_labs.shape[0])



cancer_labs.head()
class_count = cancer_labs["label"].value_counts()



print("Positive cancer scans:", class_count[1])

print("Positive cancer scans percent:", round(class_count[1] / cancer_labs.shape[0], 2) * 100)

print("Negative cancer scans:", class_count[0])

print("Negative cancer scans percent:", round(class_count[0] / cancer_labs.shape[0], 2) * 100)
train, test = train_test_split(cancer_labs, test_size=0.2, random_state=1017)



print("Cancer image training set rows:", train.shape[0])

print("Cancer image test set rows:", test.shape[0])
batch_size = 128



image_size = (96, 96)



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True

)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

    dataframe=train, 

    directory='../input/histopathologic-cancer-detection/train/', 

    x_col='id', 

    y_col='label',

    target_size=image_size,

    batch_size=batch_size,

    class_mode="binary",

    has_ext=False

)



test_generator = test_datagen.flow_from_dataframe(

    dataframe=test, 

    directory='../input/histopathologic-cancer-detection/train/', 

    x_col='id', 

    y_col='label',

    target_size=image_size,

    batch_size=batch_size,

    class_mode="binary",

    has_ext=False

)
weights_path = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model  = VGG19(include_top=False, weights= weights_path,input_shape=(96, 96, 3))



x = base_model.output

x = Flatten()(x)

x = Dense(256, activation='relu', kernel_initializer='glorot_uniform', use_bias=False)(x)

x = BatchNormalization()(x)

x = Dropout(.5)(x)

x = Dense(256, activation='relu', use_bias=False)(x)

x = Dropout(.5)(x)

predictions = Dense(1, activation = 'sigmoid')(x)



model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss="binary_crossentropy",

              optimizer="rmsprop",

              metrics=["binary_accuracy"])
from keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')



history  = model.fit_generator(generator=train_generator, 

                                         epochs=25, 

                                         steps_per_epoch=512, 

                                         validation_data=test_generator, 

                                         validation_steps=128,

                                          callbacks = [earlyStopping,mcp_save])
model.summary()
import matplotlib.pyplot as plt



def plot_history(history):

    acc = history.history["binary_accuracy"]

    val_acc = history.history["val_binary_accuracy"]

    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    x = range(1, len(acc) + 1)

    

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, "b", label="Training acc")

    plt.plot(x, val_acc, "r", label="Validation acc")

    plt.xticks(x, x)

    plt.title("Training and validation accuracy")

    plt.legend()

    

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, "b", label="Training loss")

    plt.plot(x, val_loss, "r", label="Validation loss")

    plt.xticks(x, x)

    plt.title("Training and validation loss")

    plt.legend()

    

plot_history(history=history)
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score

import matplotlib.pyplot as plt



model_probs = model.predict_generator(test_generator, steps=len(test_generator.filenames) / 128, verbose=1)
roc_auc_score(test_generator.classes, model_probs)
fpr, tpr, thresholds = roc_curve(test_generator.classes, model_probs)

i = np.arange(len(tpr)) # index for df

roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})

roc.loc[(roc.tf-0).abs().argsort()[:1]]
f= roc.loc[(roc.tf-0).abs().argsort()[:1]].thresholds

y_pred1 = np.where(model_probs > f.values[0], 1, 0)

print("F1 score is equivalent to {}".format(f1_score(test_generator.classes,y_pred1)))
from sklearn.metrics import classification_report

print(classification_report(test_generator.classes,y_pred1))