import numpy as np

import pandas as pd

import random, cv2, os

# import os, cv2, random

# import numpy as np

# import pandas as pd



import matplotlib.pyplot as plt

# from matplotlib import ticker

# import seaborn as sns

%matplotlib inline 



from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils

from sklearn.model_selection import StratifiedKFold, cross_val_score
# Will retrieve all file directories and lable them with dogs and cats.

SAMEPLE_SIZE = 200 # To expedite development, we will first sample a very small portion of all images to do programming, but this number can later be changed to scale up

TRAIN_DIRECTORY = "../input/train/"

TEST_DIRECTORY = "../input/test/"



dog_directories = [TRAIN_DIRECTORY + filename for filename in os.listdir(TRAIN_DIRECTORY) if 'dog' in filename]

cat_directories = [TRAIN_DIRECTORY + filename for filename in os.listdir(TRAIN_DIRECTORY) if 'cat' in filename]

test_directories_all = [TEST_DIRECTORY + filename for filename in os.listdir(TEST_DIRECTORY)]



# shuffle directories in case we want to have different images

random.shuffle(dog_directories)

random.shuffle(cat_directories)

random.shuffle(test_directories_all)

train_directories = dog_directories[:SAMEPLE_SIZE] + cat_directories[:SAMEPLE_SIZE]

random.shuffle(train_directories)



test_directories = test_directories_all[:25]
y_train = []



for filepath in train_directories:

    if("dog" in filepath):

        y_train.append(1)

    else:

        y_train.append(0)
ROWS = 64

COLS = 64

CHANNELS = 3

sample_image_directory = train_directories[5]

temp = cv2.imread(sample_image_directory, 1)

img = cv2.resize(temp, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

plt.imshow(img)

plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.show()
# Load all images to data after resize

ROWS = 64

COLS = 64

CHANNELS = 3





def load_images(directories):

    count = len(directories)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for (i, file) in enumerate(directories):

        temp = cv2.imread(file, 1) # 1 = load color image

        img = cv2.resize(temp, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

        data[i] = img.T

        if i > 1 and i%50 == 0: print("Loaded {} of {}.".format(i, count))

    return data

x_train = load_images(train_directories)

x_test = load_images(test_directories)
model = Sequential()



model.add(Conv2D(32, (3, 3), padding="same", input_shape=(3, ROWS, COLS), activation='relu' , data_format="channels_first"))

model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))

model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))

model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
model.summary()
class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []



    def on_batch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        

history = LossHistory()



epochs  = 10

batch_size = 16



early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        



model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,

              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])
history.losses
y_predict = model.predict(x_test)
# Function for show single image

def show_img(data, text=""):

    plt.figure(figsize=(10,5))

    plt.imshow(data.T)

    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.title(text)

    plt.show()

show_img(x_train[0])
arr = random.sample(range(0,24), 4)

y_animal = ["dog" if i > 0.5 else "cat" for i in y_predict]

for i in arr:

    show_img(x_test[i], "I'm {}% sure am a {}.".format(100*y_predict[i], y_animal[i]))