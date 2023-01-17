# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import tensorflow as tf
from tensorflow import keras
from keras.models import Model

def append_ext(fn):
    return fn+".jpg"
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
train["image_name"] = train["image_name"].apply(append_ext)
test["image_name"] = test["image_name"].apply(append_ext)
train_generator = datagen.flow_from_dataframe(dataframe=train, directory="../input/siim-isic-melanoma-classification/jpeg/train",
                                               x_col="image_name", y_col="target",target_size=(120,160),subset="training",
                                               shuffle=True,
                                               batch_size=64,
                                              class_mode="other")

val_generator = datagen.flow_from_dataframe(dataframe=train, directory="../input/siim-isic-melanoma-classification/jpeg/train",
                                               x_col="image_name", y_col="target",target_size=(120,160),subset="validation",
                                               shuffle=True,
                                               batch_size=64,
                                               class_mode="other")

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(dataframe=test,directory="../input/siim-isic-melanoma-classification/jpeg/test",
                                                x_col="image_name",y_col=None,
                                                batch_size=64,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(120,160))
from tensorflow.keras.applications import InceptionResNetV2
basemodel = InceptionResNetV2(include_top=False,weights="imagenet",input_shape=(120,160,3),pooling='avg')
headmodel = basemodel.output
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(16, activation="relu")(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(4, activation="relu")(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(1, activation="sigmoid")(headmodel)
model = Model(inputs=basemodel.input, outputs=headmodel)
basemodel.trainable = False

model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss="mse", metrics=[tf.keras.metrics.AUC()])
model.fit_generator(generator=train_generator,
                          steps_per_epoch = train_generator.n//train_generator.batch_size,
                            validation_data = val_generator,
                          validation_steps = val_generator.n//val_generator.batch_size,
                          epochs=3)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.evaluate_generator(generator=val_generator,
steps=STEP_SIZE_TEST)
test_generator.reset()
preds=model.predict_generator(test_generator)

preds
preds = pd.DataFrame(preds)
preds
df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
preds["image_name"] = df["image_name"]
preds.columns = ["target","image_name"]
column_order = ["image_name", "target"]
preds=preds.reindex(columns=column_order)
preds
preds.to_csv("sub1.csv")
