import os, cv2, random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

%matplotlib inline 



from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]



# select a subset of the data. 1000 images of cats and dogs each

train_images = train_dogs[:1000] + train_cats[:1000]

# mix it up

random.shuffle(train_images)

test_images = test_images[:25]



print("Done")
ROWS = 64

COLS = 64

CHANNELS = 3



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)



def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train = prep_data(train_images)

test = prep_data(test_images)



print("Train_shape: {}".format(train.shape))

print("Test_shape: {}".format(test.shape))
def show_cats_and_dogs(idx):

    cat = read_image(train_cats[idx])

    dog = read_image(train_dogs[idx])

    pair = np.concatenate((cat, dog), axis=1)

    plt.figure(figsize=(10,5))

    plt.imshow(pair)

    plt.show()



for img in range(0,3):

    show_cats_and_dogs(img)
labels = []

for i in train_images:

    if 'dog' in i:

        labels.append(1)

    else:

        labels.append(0)



sns.countplot(labels)

sns.plt.title('Cats and Dogs')
optimizer = RMSprop(lr=1e-4)

objective = 'binary_crossentropy'



from keras import backend as K

K.set_image_dim_ordering('th')



def catdog():

    model = Sequential()

    

    model.add(Conv2D(32, (3, 3), input_shape=(3, ROWS, COLS), activation='relu', padding='same'))

    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model



model = catdog()

model.summary()
## Callback for loss logging per epoch

class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        print("epoch done")



def run_catdog():

    history = LossHistory()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') 

    

    model.fit(train, labels, batch_size=16, epochs=10,

              validation_split=0.25, verbose=2, shuffle=True, callbacks=[history, early_stopping])

    

    predictions = model.predict(test, verbose=0)

    return predictions, history



predictions, history = run_catdog()
loss = history.losses

val_loss = history.val_losses

epochs = 10

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('VGG-16 Loss Trend')

plt.plot(loss, 'blue', label='Training Loss')

plt.plot(val_loss, 'green', label='Validation Loss')

plt.xticks(range(0,epochs)[0::2])

plt.legend()

plt.show()
for i in range(0,10):

    if predictions[i, 0] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

        

    plt.imshow(test[i].T)

    plt.show()
batch_size = 16

test_labels = []

for i in test_images:

    if 'dog' in i:

        test_labels.append(1)

    else:

        test_labels.append(0)



predictions = model.predict(test, batch_size, verbose=0)

evaluations = model.evaluate(test, test_labels, batch_size, verbose=0, sample_weight=None)

print("scalar loss: ", evaluations[0])

print("accuracy: ", evaluations[1])

model.save("test1")
from keras.models import model_from_json

from os import listdir

import pickle

model_json = model.to_json()

model_weights = model.get_weights()



pickle.dump(model_json, open('model.pkl', 'wb'))

pickle.dump(model_weights, open('model_weights.pkl', 'wb'))

os.listdir()