import os

import cv2

import zipfile

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import os

import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

import matplotlib.pyplot as plt
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:

    z.extractall(".")

    

with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:

    z.extractall(".")


# define location of dataset

folder = '/kaggle/working/train/'

for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # define filename

    filename = folder + 'dog.' + str(i) + '.jpg'

    # load image pixels

    image = plt.imread(filename)

    # plot raw pixel data

    plt.imshow(image)

    #show the figure

plt.show()
for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # define filename

    filename = folder + 'cat.' + str(i) + '.jpg'

    # load image pixels

    image = plt.imread(filename)

    # plot raw pixel data

    plt.imshow(image)

    #show the figure

plt.show()
TRAIN_DIR = "/kaggle/working/train/"

TEST_DIR="/kaggle/working/test1/"

TRAIN_SIZE = len([name for name in os.listdir(TRAIN_DIR)])

print("Number of training images:", TRAIN_SIZE)
IMAGE_WIDTH=80

IMAGE_HEIGHT=80

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
filenames = os.listdir(TRAIN_DIR)

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category.lower()=='dog':

        categories.append("dog")

    else:

        categories.append("cat")
df = pd.DataFrame({

    'filename': filenames,

    'label': categories

})
df['label']=df['label'].astype(str)
df.head()
df['label'].value_counts().plot(kind='bar')

plt.title("Number of Cats and Dogs Sample in the Dataset")

plt.show()
df['label'].value_counts()
df['label'] = df['label'].map({"dog":"1","cat":"0"})
df['label']=df['label'].astype('str')
train_df, valid_df = train_test_split(df, test_size=0.2)
train_datagen = ImageDataGenerator(    

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    rescale=1./255.,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

    )



test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    TRAIN_DIR, 

    x_col='filename',

    y_col='label',

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    class_mode='binary',

)
valid_generator = test_datagen.flow_from_dataframe(

    valid_df, 

    TRAIN_DIR, 

    x_col='filename',

    y_col='label',

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    class_mode='binary'

)
model = Sequential()



# First Set of Convolution and Pooling Layer

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(MaxPooling2D(pool_size=(2, 2)))



# Second Set of Convolution and Pooling Layer

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# Third Set of Convolution and Pooling Layer

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# Fully Connected Layer

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid')) # 2 because we have cat and dog classes
model.summary()
tf.keras.utils.plot_model(model)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),

    loss='binary_crossentropy',

    metrics = ['accuracy'])
history = model.fit_generator(train_generator,validation_data=valid_generator,steps_per_epoch=round(TRAIN_SIZE*(1.-0.2)/32),

    validation_steps=round(TRAIN_SIZE*0.2/32),epochs=20,verbose=1)
acc = history.history['accuracy']

val_acc = history.history[ 'val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs   = range(len(acc)) # Get number of epochs

plt.plot(epochs, acc)

plt.plot(epochs, val_acc)

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss)

plt.plot(epochs, val_loss)

plt.legend(['Training Loss','Validation Loss'])

plt.title('Training and validation loss')
test_filenames = os.listdir(TEST_DIR)

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]


test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    TEST_DIR, 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=32,

    shuffle=False

)
import numpy as np

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/32))

threshold = 0.5

test_df['category'] = np.where(predict > threshold, 1,0)
test_df['category'].value_counts()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)