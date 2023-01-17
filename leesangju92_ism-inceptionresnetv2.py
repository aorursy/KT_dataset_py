import numpy as np 

import pandas as pd 

import os
train = pd.read_csv("../input/invasive-species-monitoring/train_labels.csv")

train["name"] = train["name"].apply(lambda x : "../input/invasive-species-monitoring/train/" + str(x) + ".jpg")

train["invasive"] = train["invasive"].astype("str")
train.head()
from keras.preprocessing.image import ImageDataGenerator

train_image_gen = ImageDataGenerator(rescale = 1/255, validation_split=0.022, rotation_range=270, horizontal_flip=True)

batch_size = 6

image_size = 800

train_generator = train_image_gen.flow_from_dataframe(dataframe=train, batch_size=batch_size, target_size=(image_size,image_size), x_col="name", y_col="invasive", subset="training", class_mode="binary")

val_generator = train_image_gen.flow_from_dataframe(dataframe=train, batch_size=batch_size, target_size=(image_size,image_size), x_col="name", y_col="invasive", subset="validation", class_mode="binary")
from keras import Sequential

from keras.layers import Dense, Dropout

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.mobilenet_v2 import MobileNetV2 

model = Sequential()

model.add(InceptionResNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(image_size, image_size, 3)))

model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))
model.summary()
from keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.0003, decay=2e-7), loss='binary_crossentropy',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

modelcheck = ModelCheckpoint("best.h5", save_best_only=True)

model.fit_generator(train_generator, steps_per_epoch=int(np.ceil(train_generator.n/batch_size)), validation_data = val_generator, validation_steps = int(np.ceil(val_generator.n/batch_size)), epochs=6, callbacks=[modelcheck])
sub = pd.read_csv("../input/invasive-species-monitoring/sample_submission.csv")

test = pd.DataFrame({"name": sub["name"].apply(lambda x : "../input/invasive-species-monitoring/test/" + str(x) + ".jpg")})

test_gen = ImageDataGenerator(rescale = 1./255)

test_generator = test_gen.flow_from_dataframe(dataframe = test, shuffle=False, x_col = "name", y_col = None, class_mode=None, batch_size = batch_size, target_size = (image_size, image_size))

model.load_weights("best.h5")

preds = model.predict_generator(test_generator, steps=int(np.ceil(test_generator.n/batch_size)), workers = 2)

sub["invasive"] = preds

sub.to_csv("submission.csv", index=False)