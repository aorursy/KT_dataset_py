# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



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

import random

import os

FAST_RUN = False

IMAGE_WIDTH=150

IMAGE_HEIGHT=150

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
train=pd.read_csv('/kaggle/input/nnfl-cnn-lab2/upload/train_set.csv')
train.head()

train['label']=train['label'].astype('str')

train['label'].value_counts().plot.bar()
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



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



model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction,ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
filenames=os.listdir("/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/")
filenames[0]
image = load_img("/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/"+filenames[0])

if np.shape(np.array(image))!=(150,150,3):

  print('oh no')
count=0

for i in range(len(filenames)):

  image = load_img("/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/"+filenames[i])

  if np.shape(np.array(image))!=(150,150,3):

    count+=1

    print(np.shape(np.array(image)))

print(count)
batch_size=15
train_df, validate_df = train_test_split(train, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



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
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
epochs=30

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
test_filenames = os.listdir("/kaggle/input/nnfl-cnn-lab2/upload/test_images/test_images")

test_df = pd.DataFrame({

    'image_name': test_filenames

})

nb_samples = test_df.shape[0]
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
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['label'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['label'] = test_df['label'].replace(label_map)
submission_df = test_df.copy()

submission_df.to_csv('submission.csv', index=False)
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