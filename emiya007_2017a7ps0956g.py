# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(42)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import random

FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
filenames = os.listdir('/kaggle/input/nnfl-lab-1/training/training')

categories = []

for filename in filenames:

    category = filename.split('_')[0]

    if category == 'chair':

        categories.append(0)

    if category == 'kitchen':

        categories.append(1)

    if category == 'knife':

        categories.append(2)

    if category == 'saucepan':

        categories.append(3)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.head()
test_df=pd.read_csv("/kaggle/input/nnfl-lab-1/sample_sub.csv")
test_df.head()
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization 

from tensorflow.python.keras import Sequential 



model = Sequential() 

model.add(Conv2D(64, (3, 3), activation='relu', padding='same',input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))) 

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(64, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(128, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(128, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25)) 



model.add(Conv2D(256, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(256, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Flatten()) 

model.add(Dense(256, activation='relu')) 

model.add(BatchNormalization()) 

model.add(Dropout(0.5))



model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
df['category'] = df['category'].astype(str)
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=21)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    '/kaggle/input/nnfl-lab-1/training/training/', 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    '/kaggle/input/nnfl-lab-1/training/training/', 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
epochs=5 if FAST_RUN else 40

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.save_weights("model.h5")
model.summary()
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    '/kaggle/input/nnfl-lab-1/testing/testing/', 

    x_col='id',

    y_col='label',

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
nb_samples = test_df.shape[0]

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['label'] = np.argmax(predict, axis=-1)
test_df.head()
submission_df = test_df.copy()

submission_df.to_csv('submission29.csv', index=False)
df=submission_df

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)