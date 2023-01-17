import os

import cv2



import numpy as np

import pandas as pd



from matplotlib import pyplot as plt



from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer



from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras_preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers, optimizers

from keras.models import Sequential
INPUT = "/kaggle/input/imet-2020-fgvc7/"

TRAIN_DIR = "/kaggle/input/imet-2020-fgvc7/train/"

TEST_DIR = "/kaggle/input/imet-2020-fgvc7/test/"



EPOCHS = 6

BATCH_SIZE = 512

IM_SIZE = 128
train_df = pd.read_csv(INPUT + "train.csv")

test_df = pd.read_csv(INPUT + "sample_submission.csv")

labels_df = pd.read_csv(INPUT + "labels.csv")
# adding "png" to the image id

test_df["id"] = test_df["id"] + ".png" 

train_df['id'] = train_df["id"] + ".png"

train_df.head(2)
# labels = labels_df.attribute_id.to_list()
train_df["attribute_ids"] = train_df["attribute_ids"].apply(lambda x:x.split())

print(train_df.shape)

train_df.head(3)
datagen = ImageDataGenerator(width_shift_range=0.1,

                             height_shift_range=0.1,

                             zoom_range=0.2,

                             rescale=1/255. )



test_datagen = ImageDataGenerator(rescale=1./255.)
train_generator = datagen.flow_from_dataframe(

                                            dataframe=train_df,

                                            directory=TRAIN_DIR,

                                            x_col="id",

                                            y_col="attribute_ids",

                                            batch_size=BATCH_SIZE,

                                            seed=42,

                                            shuffle=True,

                                            class_mode="categorical",

#                                             classes=labels,

                                            target_size=(IM_SIZE,IM_SIZE))
# valid_generator = test_datagen.flow_from_dataframe(

#                                                  dataframe=val_x,

#                                                  directory=TRAIN_DIR,

#                                                  x_col="id",

#                                                  y_col="attribute_ids",

#                                                  batch_size=BATCH_SIZE,

#                                                  seed=42,

#                                                  shuffle=True,

#                                                  class_mode="categorical",

# #                                                  classes=labels,

#                                                  target_size=(IM_SIZE,IM_SIZE))

test_generator = test_datagen.flow_from_dataframe(

                                                dataframe=test_df,

                                                directory=TEST_DIR,

                                                x_col="id",

                                                batch_size=1,

                                                seed=42,

                                                shuffle=False,

                                                class_mode=None,

                                                target_size=(IM_SIZE,IM_SIZE))
model = Sequential()

inputShape = (IM_SIZE, IM_SIZE, 3)

chanDim = -1



# first CONV => RELU => CONV => RELU => POOL layer set

model.add(Conv2D(32, (3, 3), padding="same",

    input_shape=inputShape))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(32, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# second CONV => RELU => CONV => RELU => POOL layer set

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# third CONV => RELU => CONV => RELU => POOL layer set

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# first (and only) set of FC => RELU layers

model.add(Flatten())

model.add(Dense(512))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))



# softmax classifier

model.add(Dense(3471, activation='sigmoid')) 



model.compile(optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])
def build_lrfn(lr_start=0.00001, lr_max=0.000075, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    

    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn



lrfn = build_lrfn()

lr_schedule = LearningRateScheduler(lrfn, verbose=1)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

# STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

#                     validation_data=valid_generator,

#                     validation_steps=STEP_SIZE_VALID,

                    callbacks=[lr_schedule],

                    epochs=EPOCHS

                   )
test_generator.reset()

pred=model.predict_generator(test_generator,

steps=STEP_SIZE_TEST,

verbose=1)

pred_bool = (pred >0.2)

predictions=[]

labels = train_generator.class_indices

labels = dict((v,k) for k,v in labels.items())

for row in pred_bool:

    l=[]

    

    for index,cls in enumerate(row):

        if cls:

            l.append(labels[index])

    predictions.append(" ".join(l))

    

filenames=test_generator.filenames



results = pd.DataFrame({"id":filenames,"attribute_ids":predictions})

results["id"] = results["id"].apply(lambda x:x.split(".")[0])

results.to_csv("submission.csv",index=False)