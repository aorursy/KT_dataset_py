# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/nnfl-cnn-lab2/upload/train_set.csv")

df["label"] = df["label"].replace({0: 'A', 1: 'B',2:'C',3:'D',4:'E',5:'F'})

df.head()
FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D, Dropout, Flatten, Dense, Activation,BatchNormalization
df['label'].value_counts().plot.bar()
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



#m



model.add(Flatten())



#m

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

#from keras import optimizers

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=14)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5,min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)

#train_df['label'].value_counts().plot.bar()

validate_df['label'].value_counts().plot.bar()
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)

batch_size=15

train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
epochs=3 if FAST_RUN else 14

total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.save_weights("model1.h5")
test_df=pd.read_csv("/kaggle/input/nnfl-cnn-lab2/upload/sample_submission.csv")

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "/kaggle/input/nnfl-cnn-lab2/upload/test_images/test_images", 

    x_col='image_name',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
test_df.head()
nb_samples = test_df.shape[0]

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['label'] = np.argmax(predict, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['label'] = test_df['label'].replace(label_map)

test_df["label"] = test_df["label"].replace({'A': 0, 'B': 1,'C':2,'D':3,'E':4,'F':5})
submission_df = test_df.copy()

submission_df.to_csv('submission11.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(submission_df, title = "Download CSV file", filename = "data.csv"):

    csv = submission_df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(submission_df)