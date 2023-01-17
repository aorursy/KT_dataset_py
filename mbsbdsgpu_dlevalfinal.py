# Please ensure that tensorflow version 1.13 is installed. Otherwise the code won't work
# !pip install tensorflow==1.13.0rc1
# import tensorflow as tf
# print(tf.__version__)
import numpy as np 
import pandas as pd 
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization

import h5py
def append_ext(fn):
    return fn+".jpg"

train_df=pd.read_csv('../input/dog-breed-identification/labels.csv')
test_df=pd.read_csv('../input/dog-breed-identification/sample_submission.csv')


train_df["id"]=train_df["id"].apply(append_ext)
test_df["id"]=test_df["id"].apply(append_ext)
import matplotlib.pyplot as plt
plt.figure(figsize=(13, 6))
train_df['breed'].value_counts().plot(kind='bar')
plt.show()
num_classes = 120
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

TL_model = Sequential()
TL_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
TL_model.add(Dense(512))
TL_model.add(Activation('relu'))
TL_model.add(Dropout(0.5))
TL_model.add(Dense(num_classes, activation='softmax'))

TL_model.layers[0].trainable = False

TL_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
TL_model.summary()
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                             rescale=1./255.,
                             horizontal_flip=True,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             validation_split=0.2)
train_batch_size = 10
train_generator=datagen.flow_from_dataframe(
                        dataframe=train_df,
                        directory="../input/dog-breed-identification/train/",
                        x_col="id",
                        y_col="breed",
                        has_ext=False,
                        subset="training",
                        batch_size=train_batch_size,
                        seed=50,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(image_size, image_size))
valid_generator=datagen.flow_from_dataframe(
                        dataframe=train_df,
                        directory="../input/dog-breed-identification/train/",
                        x_col="id",
                        y_col="breed",
                        has_ext=False,
                        subset="validation",
                        batch_size=1,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(image_size, image_size))
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
                            dataframe=test_df,
                            directory="../input/dog-breed-identification/test/",
                            x_col="id",
                            y_col=None,
                            has_ext=False,
                            batch_size=1,
                            seed=42,
                            shuffle=False,
                            class_mode=None,
                            target_size=(image_size, image_size))
train_step_size = train_generator.n
validation_step_size = valid_generator.n

TL_model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_step_size,
                    validation_data=valid_generator,
                    validation_steps=validation_step_size,
                    epochs=25
)
TL_model.evaluate_generator(generator=valid_generator)
test_generator.reset()
pred = TL_model.predict_generator(test_generator,verbose=1)
labels = (train_generator.class_indices)
labels = list(labels.keys())
df = pd.DataFrame(data=pred,
                 columns=labels)

columns = list(df)
columns.sort()
df = df.reindex(columns=columns)

test_df=pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
filenames = test_df["id"]
df["id"]  = filenames

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
# df.head(5)
df.to_csv("submission2.csv",index=False)
