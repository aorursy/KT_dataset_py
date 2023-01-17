import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



print(os.listdir("../input"))

%matplotlib inline

%reload_ext autoreload

%autoreload 2



!pip install -U EfficientNetB3









base_dir="../input/"

train_dir=os.path.join(base_dir,"train/train")

test_dir=os.path.join(base_dir,"test/test")

print("Training images : \n{}".format(os.listdir(train_dir)[:10]), end='\n\n')

print("Testing images : \n{}".format(os.listdir(test_dir)[:10]))

testing_dir=os.path.join(base_dir,"test")
train_dataframe = pd.read_csv("../input/train.csv")

train_dataframe["has_cactus"] = np.where(train_dataframe["has_cactus"] == 1, "yes", "no")

print(train_dataframe.head())
import keras

from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import applications

from efficientnet import EfficientNetB3
train_datagen = ImageDataGenerator(

    rescale=1/255,

    validation_split=0.10,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = train_dataframe,

    directory = train_dir,

    x_col="id",

    y_col="has_cactus",

    target_size=(75,75),

    subset="training",

    batch_size=250,

    shuffle=True,

    class_mode="binary"

)



valid_generator = train_datagen.flow_from_dataframe(

    dataframe = train_dataframe,

    directory = train_dir,

    x_col="id",

    y_col="has_cactus",

    target_size=(75,75),

    subset="validation",

    batch_size=125,

    shuffle=True,

    class_mode="binary"

)
test_datagen = ImageDataGenerator(

    rescale=1/255

)



test_generator = test_datagen.flow_from_directory(

    testing_dir,

    target_size=(75,75),

    batch_size=1,

    shuffle=False,

    class_mode=None

)
pretrained_net = EfficientNetB3(

    input_shape=(75,75,3),

    include_top=False,

    pooling='max',

)

model = Sequential()

model.add(pretrained_net)

model.add(Dense(units = 120, activation='relu'))

model.add(Dense(units = 84, activation = 'relu'))

model.add(Dense(units = 1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(

    train_generator,

    epochs = 20,

    steps_per_epoch = 63,

    validation_data = valid_generator,

    validation_steps = 14

)

acc, loss = history.history['acc'], history.history['loss']

val_acc, val_loss = history.history['val_acc'], history.history['val_loss']



epochs = len(acc)



import matplotlib.pyplot as plt



plt.plot(range(epochs), acc, color='red', label='Training Accuracy')

plt.plot(range(epochs), val_acc, color='green', label='Validation Accuracy')

plt.legend()

plt.title('Accuracy over Training')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()



plt.plot(range(epochs), loss, color='red', label='Training Loss')

plt.plot(range(epochs), val_loss, color='green', label='Validation Loss')

plt.legend()

plt.title('Loss over Training')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
preds = model.predict_generator(

    test_generator,

    steps=len(test_generator.filenames)

)
image_ids = [name.split('/')[-1] for name in test_generator.filenames]

preds = preds.flatten()

data = {'id': image_ids, 'has_cactus':preds} 

submission = pd.DataFrame(data)

print(submission.head())
submission.to_csv("submission.csv", index=False)