# Cancer Detection project, By BEKKAR Abdellatif

# Aim is to familiarity with the use of CNNs with tensorflow for image classification
# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from time import time

import seaborn as sns

import plotly.graph_objects as go



from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator





# import the necessary packages

from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras import backend as K

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#Total Samples Available

print('Train Images = ',len(os.listdir('../input/histopathologic-cancer-detection/train')))

print('Test Images = ',len(os.listdir('../input/histopathologic-cancer-detection/test')))
df = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv',dtype=str)

print(df.head())
print('Number of image : ', len(df))

img = plt.imread("../input/histopathologic-cancer-detection/train/"+df.iloc[0]['id']+'.tif')

print('Images shape', img.shape)
for i in range(5):

    img = plt.imread("../input/histopathologic-cancer-detection/train/"+df.iloc[i]['id']+'.tif')

    print(df.iloc[i]['label'])

    plt.imshow(img)

    plt.show()
# Descriptive Analytics for given Dataset



print(df.label.value_counts())
fig = plt.figure(figsize = (6,6)) 

ax = sns.countplot(df.label).set_title('Label Counts', fontsize = 18)

plt.annotate(df.label.value_counts()[0],

            xy = (0,df.label.value_counts()[0] + 2000),

            va = 'bottom',

            ha = 'center',

            fontsize = 12)

plt.annotate(df.label.value_counts()[1],

            xy = (1,df.label.value_counts()[1] + 2000),

            va = 'bottom',

            ha = 'center',

            fontsize = 12)

plt.ylim(0,150000)

plt.ylabel('Count', fontsize = 16)

plt.xlabel('Labels', fontsize = 16)

plt.show()
labels = ["No Cancer - 0", "Cancer - 1"]

values = df.label.value_counts()



d = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=["rgb(0, 76, 153)","rgb(255, 158, 60)"])])

d.show()
#add .tif to ids in the dataframe to use flow_from_dataframe

df["id"]=df["id"].apply(lambda x : x +".tif")

df.head()
train_path = '../input/histopathologic-cancer-detection/train'

valid_path = '../input/histopathologic-cancer-detection/train'
train_datagen = ImageDataGenerator(validation_split=0.20,

                          rescale=1/255.0)
train_generator=train_datagen.flow_from_dataframe(

    dataframe=df,

    directory=train_path,

    x_col="id",

    y_col="label",

    subset="training",

    batch_size=64,

    shuffle=True,

    class_mode="binary",

    target_size=(96,96))
valid_generator=train_datagen.flow_from_dataframe(

    dataframe=df,

    directory=valid_path,

    x_col="id",

    y_col="label",

    subset="validation",

    batch_size=64,

    shuffle=True,

    class_mode="binary",

    target_size=(96,96))

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (96, 96, 3)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(3,3)))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(3,3)))



model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'))

model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=(3,3)))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy'])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

EPOCHS=20


earlystopper = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True)

reducel = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.1)





history = model.fit_generator(generator=train_generator, 

                    steps_per_epoch=STEP_SIZE_TRAIN, 

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                   callbacks=[reducel, earlystopper])

def show_final_history(history):

    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('accuracy')

    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")

    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")

    ax[0].legend()

    ax[1].legend()
show_final_history(history)

print("Validation Accuracy: " + str(history.history['val_accuracy'][-1:]))