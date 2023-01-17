import json

import os

from PIL import Image

from glob import glob

from zipfile import ZipFile

import pandas as pd

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.models import Model, Input, load_model

from keras.layers import Dense, GlobalAveragePooling2D

from keras.applications.mobilenetv2 import MobileNetV2

from keras import backend as K

from matplotlib import pyplot as plt

from matplotlib.image import imread

import cv2

import numpy as np
def process_csv(dataframe: pd.DataFrame, image_column_name: str,

                label_column_name: str,

                folder_with_images: str) -> pd.DataFrame:

    """This function process Pandas DataFrame, which contains image filenames

    and their corresponding labels.



    Args:

        dataframe: Pandas DataFrame object. It should consist of 2 columns

        image_column_name: The name of the column containing the image

            filenames

        label_column_name: The name of the column containing the image

            labels

        folder_with_images: Folder with images



    Returns:

        dataframe: processed DataFrame with full paths to images

    """

    dataframe[image_column_name] = dataframe[image_column_name].apply(

        lambda x: f"{folder_with_images}{x}.png")

    dataframe[label_column_name] = dataframe[label_column_name].astype('str')

    return dataframe
train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   rotation_range=15,

                                   width_shift_range=0.1,

                                   height_shift_range=0.1,

                                   shear_range=0.01,

                                   zoom_range=[0.9, 1.25],

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   fill_mode='reflect',

                                   data_format='channels_last',

                                   brightness_range=[0.5, 1.5],

                                   validation_split=0.3)
train_csv = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")

train_csv = process_csv(

    dataframe=train_csv,

    image_column_name="id_code",

    label_column_name="diagnosis",

    folder_with_images="/kaggle/input/aptos2019-blindness-detection/train_images/")



train_generator = train_datagen.flow_from_dataframe(

    dataframe=train_csv, x_col="id_code", y_col="diagnosis", subset="training",

    batch_size=16, target_size=(224, 224))

val_generator = train_datagen.flow_from_dataframe(

    dataframe=train_csv, x_col="id_code", y_col="diagnosis",

    subset="validation", batch_size=16, target_size=(224, 224))
from keras.models import Model, Input

from keras.layers import Dense, GlobalAveragePooling2D

from keras.applications.mobilenetv2 import MobileNetV2

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Average, Input





def create_model():

    input_tensor = Input((224, 224, 3))

    outputs = []

    

    mobilenet_model = MobileNetV2(weights=None, input_shape=(224, 224, 3),include_top=False, alpha=1.4)                          

    mobilenet_model.load_weights("/kaggle/input/mobilenet-v2-keras-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5")

    

    VGG16_model = VGG16(weights=None, input_shape = (224,224,3), include_top=False)

    VGG16_model.load_weights("/kaggle/input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

    #InceptionV3_model = InceptionV3(weights=None, input_shape=(224,224,3),include_top = False)

    #InceptionV3_model.load_weights("/kaggle/input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

    pretrained_models = [

        mobilenet_model,VGG16_model

    ]

    for model in pretrained_models:

        curr_output = model(input_tensor)

        curr_output = GlobalAveragePooling2D()(curr_output)

        curr_output = Dense(1024, activation="relu")(curr_output)

        outputs.append(curr_output)

    output_tensor = Average()(outputs)

    output_tensor = Dense(5, activation="softmax")(output_tensor)



    model = Model(input_tensor, output_tensor)

    return model
class LRFinder(Callback):



    def __init__(self, start_learning_rate, multiplier):

        self.multiplier = multiplier

        self.start_learning_rate = start_learning_rate

        self.curr_train_step_number = 1

        self.learning_rate = start_learning_rate

        self.all_learning_rates = []

        self.all_loss_values = []

        super().__init__()



    def on_batch_end(self, batch, logs=None):

        self.learning_rate *= self.multiplier

        K.set_value(self.model.optimizer.lr, self.learning_rate)

        self.curr_train_step_number += 1

        self.all_learning_rates.append(self.learning_rate)

        self.all_loss_values.append(logs.get('loss'))
start_lr = 1e-10

end_lr = 1
lrfinder_generator = ImageDataGenerator(rescale=1./255, validation_split=0).flow_from_dataframe(

    dataframe=train_csv, x_col="id_code", y_col="diagnosis", subset="training",

    batch_size=8, target_size=(224, 224))

lrfinder_callback = LRFinder(1e-10, multiplier=(end_lr / start_lr) ** (1 / len(lrfinder_generator))) 

lrfinder_model = create_model()

lrfinder_model.compile(optimizer=Adam(start_lr),

                       loss="categorical_crossentropy", metrics=["accuracy"])
lrfinder_model.fit_generator(generator=lrfinder_generator,

                             steps_per_epoch=len(lrfinder_generator),

                             epochs=1,

                             callbacks=[lrfinder_callback])
def exponentionaly_weighted_average(data: list, beta: float) -> list:

    processed_data = [data[0]]

    for value in data[1:]:

        processed_data.append(beta * value + (1 - beta) * processed_data[-1])

    return processed_data
beta_values = [0.9, 0.5, 0.3, 0.1, 0.05]

losses = lrfinder_callback.all_loss_values

learning_rates = lrfinder_callback.all_learning_rates
plt.figure(1, figsize=(10, 5))

plt.xscale('log')

plt.xticks([10**(-i) for i in range(11)])

plt.plot(learning_rates, losses)

plt.xlabel("Learning rate (log scale, without EWMA)")

plt.ylabel("Loss")

plt.show()



for idx, beta_value in enumerate(beta_values):

    plt.figure(idx + 2, figsize=(10, 5))

    plt.xscale('log')

    plt.xticks([10**(-i) for i in range(11)])

    plt.plot(learning_rates, exponentionaly_weighted_average(losses, beta_value))

    plt.xlabel(f"Learning rate (log scale, with EWMA, beta={beta_value})")

    plt.ylabel("Loss")

    plt.show()
#callbacks = [

#    ModelCheckpoint(

#        "best_weights.h5",

#        monitor='val_acc',

#        verbose=1, save_best_only=True),

#    EarlyStopping(monitor='val_acc', patience=5)

#]

callbacks = [ModelCheckpoint("best_weights.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
model = create_model()

model.compile(optimizer=Adam(1e-4),

              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(generator=train_generator,

                    steps_per_epoch=len(train_generator),

                    validation_data=val_generator,

                    validation_steps=len(val_generator),

                    epochs=50,

                    callbacks=callbacks)
model = load_model("best_weights.h5")
test_csv = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")

predicted_csv = pd.DataFrame(columns=["id_code", "diagnosis"])



for id_code in test_csv["id_code"]:

    filename = f"/kaggle/input/aptos2019-blindness-detection/test_images/{id_code}.png"

    img = imread(filename)

    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    img = np.expand_dims(img, 0)

    prediction = int(np.argmax(model.predict(img)[0]))

    predicted_csv = predicted_csv.append(

        {'id_code':id_code ,"diagnosis": prediction}, ignore_index=True)



with open("submission.csv", "w") as f:

    f.write(predicted_csv.to_csv(index=False))