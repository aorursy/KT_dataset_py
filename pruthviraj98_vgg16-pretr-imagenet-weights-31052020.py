# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def append_this(fn):
    return fn+".png"

df=pd.read_csv('/kaggle/input/rsna-bone-age/boneage-training-dataset.csv', dtype=str)
df_t=pd.read_csv("/kaggle/input/rsna-bone-age/boneage-test-dataset.csv",dtype=str)

df["id"]=df["id"].apply(append_this)
df['Boneage'] = df['boneage'].astype(int)
del(df['boneage'])
df_t["Case ID"]=df_t["Case ID"].apply(append_this)
del(df['male'])
del(df_t['Sex'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

image_size=256

data_gen=ImageDataGenerator(
                height_shift_range=0.2,
                width_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                preprocessing_function=preprocess_input,
                zoom_range=0.2,
                validation_split=0.20
            )

data_gen_test=ImageDataGenerator(
                preprocessing_function=preprocess_input,
            )
train_generator=data_gen.flow_from_dataframe(
    dataframe=df,
    directory="../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/",
    x_col="id",
    y_col="Boneage",
    subset="training",
    batch_size=10,
    seed=42,
    shuffle=True,
    target_size=(image_size,image_size),
    class_mode='raw'
    )

validation_generator=data_gen.flow_from_dataframe(
    dataframe=df,
    directory="../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/",
    x_col="id",
    y_col="Boneage",
    subset="validation",
    batch_size=10,
    seed=42,
    shuffle=True,
    target_size=(image_size,image_size),
    class_mode='raw'
)

test_generator=data_gen_test.flow_from_dataframe(
    dataframe=df_t,
    directory="../input/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/",
    x_col="Case ID",
    y_col=None,
    seed=42,
    target_size=(image_size,image_size),
    class_mode=None
)
import cv2
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip (x_val, y_val, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
    c_ax.set_title('%2.0f months' % (c_y))
    c_ax.axis('off')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
model=Sequential()
model.add(VGG16(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
model.layers[0].trainable=False
model.compile(loss='mse', optimizer='adam', metrics=['MeanSquaredError'])
model.summary()
model.fit_generator(train_generator,
                   validation_data=validation_generator,
                   epochs=12,
                   )
y_pred=model.predict_generator(test_generator)
preds=y_pred.flatten()
import csv
df_temp=pd.read_csv("/kaggle/input/rsna-bone-age/boneage-test-dataset.csv")
filenames=df_temp['Case ID']
results=pd.DataFrame({"Filename":filenames,
                      "Predictions": preds})
results.to_csv("predictions.csv",index=False)