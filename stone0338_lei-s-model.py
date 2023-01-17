import numpy as np

import pandas as pd

import os

import cv2

import time

import gc

from matplotlib.pyplot import imshow

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

# from keras import layers

from keras.layers import Dropout, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Concatenate

from keras.models import Model, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.initializers import glorot_uniform

from keras import optimizers

from keras import regularizers

from keras.callbacks import ModelCheckpoint

from keras.applications.densenet import DenseNet201

from IPython.display import FileLink

from IPython.display import FileLinks

os.listdir("../input")
def read_image_test_lw(img_id):

    path = "../input/test/" + img_id + ".tif"

    img = cv2.imread(path)/ 255

    return img
def model_f(input_shape):

    X_input = Input(input_shape)

    X = DenseNet201()(X_input)

    X = Dropout(0.5)(X)

    X = Dense(1, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X)

    return model
df_data = pd.read_csv('../input/train_labels.csv')

batch_size = 32

epochs = 8

random_state = 5

model_name = "model_best_1"

size_original = 96
path_data = "../input/"

X_train_index, X_val_index, y_train, y_val = train_test_split(df_data['id'].values, df_data['label'].values, test_size=0.2, random_state=random_state)

df_train = pd.DataFrame({"id": X_train_index + ".tif", "label":y_train.astype(str)})

df_val = pd.DataFrame({"id": X_val_index + ".tif", "label":y_val.astype(str)})



train_steps = len(X_train_index)//batch_size

val_steps = len(X_val_index)//batch_size



df_test = pd.read_csv('../input/sample_submission.csv')

df_test['id'] = df_test['id'] + '.tif'
train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True)

train_generator = train_datagen.flow_from_dataframe(

        df_train,

        directory = path_data + "train/",

        x_col = "id",

        y_col = "label",

        target_size=(size_original, size_original),

        batch_size=batch_size,

        class_mode='binary')



val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(

        df_val,

        directory = path_data + "train/",

        x_col = "id",

        y_col = "label",

        target_size=(size_original, size_original),

        batch_size=batch_size,

        class_mode='binary')



test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = val_datagen.flow_from_dataframe(

        df_test,

        directory = path_data + "test/",

        x_col = "id",

        y_col = "label",

        target_size=(size_original, size_original),

        batch_size=256,

        shuffle=False,

        class_mode=None)
model_lw = model_f(input_shape=(size_original,size_original,3))

model_lw.compile(optimizer = optimizers.Adam(lr=0.0001), loss = "binary_crossentropy", metrics = ["accuracy"])
filepath = model_name + ".h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

train_history = model_lw.fit_generator(train_generator,

                                    epochs=epochs,

                                    steps_per_epoch = train_steps,

                                    validation_data = val_generator,

                                    validation_steps = val_steps,

                                    shuffle = True,

                                    callbacks=[checkpoint])



print(str(train_history.history), file = open("Model_Details.txt", "w"))
model_lw.load_weights(model_name + ".h5")
predict = model_lw.predict_generator(test_generator, steps = len(test_generator), workers=0, verbose=1)

df_test = pd.read_csv('../input/sample_submission.csv')

df_test['label'] = predict

df_test.to_csv(model_name + "_result.csv", index=False)

df_test.head



# time_start = time.time()

# print("--- %s minutes ---" % ((time.time() - time_start)/60))
FileLink("model_best_1_result.csv")
FileLink("model_best_1.h5")
FileLink("Model_Details.txt")