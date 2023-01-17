# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



        

        

# import numpy as np

# import pandas as pd 

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img

# from keras.utils import to_categorical

# from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

# import random

# import os

# from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# #print(os.listdir("../content"))



# # Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
df = pd.read_csv("../input/nnfl-cnn-lab2/upload/train_set.csv", sep=",")
df.head()
df['label'].value_counts().plot.bar()
image = load_img("../input/nnfl-cnn-lab2/upload/train_images/train_images/7.jpg")

plt.imshow(image)
df['label'].head()
df['label']=df['label'].astype(str)

df['label'].head()
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['label'].value_counts().plot.bar()
validate_df['label'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
print(total_train)

print(total_validate)
train_df.head()
test_filenames = os.listdir("../input/nnfl-cnn-lab2/upload/test_images/test_images/")

test_df = pd.DataFrame({

    'image_name': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/nnfl-cnn-lab2/upload/test_images/test_images/", 

    x_col='image_name',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
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

    "../input/nnfl-cnn-lab2/upload/train_images/train_images/", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/nnfl-cnn-lab2/upload/train_images/train_images/", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



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



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax')) 



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
epochs=10 if FAST_RUN else 20

history = model.fit(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

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
# model.load_weights("model_1.h5")
predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['label'] = np.argmax(predict, axis=-1)
test_df['label'].value_counts().plot.bar()
sample_test = test_df.head(18)

sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['image_name']

    category = row['label']

    img = load_img("../input/nnfl-cnn-lab2/upload/test_images/test_images/"+filename, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
test_df.head()
submission_df = test_df.copy()

#submission_df.drop(['image_name', 'label'], axis=1, inplace=True)

submission_df.to_csv('submission_try.csv', index=False)
submission_df.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = submission_df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)