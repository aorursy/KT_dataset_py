import pandas as pd

import numpy as np

import json

import math

import tqdm

import glob

from IPython.display import FileLink, FileLinks



import sklearn.preprocessing



from PIL import Image as img



import matplotlib.pyplot as plt



import keras

from keras.preprocessing.image import ImageDataGenerator
# train

train_df = pd.read_csv("../input/2019-3rd-ml-month-with-kakr/train.csv", engine="python")

train_df["class"] = train_df["class"].astype("str")



# class(extract automaker)

class_df = pd.read_csv("../input/2019-3rd-ml-month-with-kakr/class.csv")

class_df["id"] = class_df["id"].astype("str")

class_df["automaker"] = class_df["name"].str.split(n=1, expand=True)[0]



train_df = train_df.merge(class_df, how="inner", left_on="class", right_on="id")

train_df.drop(columns="id", inplace=True)

display(train_df.head())
# test

test_df = pd.read_csv("../input/2019-3rd-ml-month-with-kakr/test.csv")
# hyperparameter

batch_size = 32

img_size = (256, 256)

channels = (3, )

model_type = "xception"

if model_type == "xception":

    preprocess_inputs = keras.applications.xception.preprocess_input

elif model_type == "Resnet":

    preprocess_inputs = keras.applications.resnet50.preprocess_input
train_datagen = ImageDataGenerator(horizontal_flip=True,

                                   vertical_flip=True, 

                                   zoom_range=.1, preprocessing_function=preprocess_inputs)



train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, directory="../input/train", 

                                                    x_col="img_file", y_col="class", 

                                                    class_mode="categorical", color_mode="rgb", 

                                                    batch_size=batch_size, target_size=img_size)
# model.reset_states()
if model_type == "xception":

    xception = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=img_size+channels, pooling=None, classes=196)

    model = keras.models.Sequential()



    model.add(xception)

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dropout(rate=.3))



    model.add(keras.layers.Dense(5, activation='sigmoid'))



    model.add(keras.layers.Dense(units=len(train_generator.class_indices), activation="softmax", kernel_initializer='he_normal'))



    model.compile(optimizer="RMSprop", loss="categorical_crossentropy", metrics=["acc"])

    

elif model_type == "Resnet":

    Resnet = keras.applications.resnet50.ResNet50(include_top=False, input_shape=img_size+channels)

    model = keras.models.Sequential()



    model.add(Resnet)

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dropout(rate=.3))



    model.add(keras.layers.Dense(5, activation='sigmoid'))



    model.add(keras.layers.Dense(units=len(train_generator.class_indices), activation="softmax", kernel_initializer='he_normal'))



    model.compile(optimizer="RMSprop", loss="categorical_crossentropy", metrics=["acc"])
print(model_type)
%%time

model.fit_generator(train_generator, epochs=200, steps_per_epoch=300, verbose=0)
plt.plot(model.history.history["acc"])

plt.show()
test_datagen = ImageDataGenerator(horizontal_flip=False,

                                  vertical_flip=False, 

                                  zoom_range=.0, preprocessing_function=preprocess_inputs)



test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory="../input/test", 

                                                  x_col="img_file", y_col=None,

                                                  class_mode=None, color_mode="rgb", 

                                                  batch_size=batch_size, target_size=img_size)
%%time

result = model.predict_generator(generator=test_generator, steps=math.ceil(len(test_df)/batch_size), verbose=0)
result = np.argmax(result, axis=1)

temp = pd.DataFrame({"index" : test_generator.index_array, "class" : result})
label = train_generator.class_indices

label = pd.DataFrame.from_dict(label, orient='index').reset_index().rename(columns={0:"class"})
result = temp.merge(label, left_on="class", right_on="class")
submission = test_df.merge(result, left_index=True, right_on="index_x")[["img_file", "class"]]
submission.to_csv("submission.csv", index=False)