import warnings

warnings.filterwarnings("ignore")

import os, cv2, random

import numpy as np

import pandas as pd

%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib import ticker

import seaborn as sns

%matplotlib inline 



from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils, to_categorical
path = '/kaggle/input/flower-recognition/flower_recognition/'
train_label = pd.read_csv(path + 'train.csv')

train_label.head()
unique_train_label = np.unique(train_label['category'].values)

unique_train_label
plt.figure(figsize=(33, 15))

sns.countplot(train_label['category'])

plt.show()
# Displaying one image to check

img=mpimg.imread(path + 'train/3261.jpg')

imgplot = plt.imshow(img)

plt.show()
# train image names along with paths

train_image_name = [path+'train/'+str(each)+'.jpg' for each in train_label['image_id'].values.tolist()]

train_image_name[:5]
test_label = pd.read_csv(path + 'test.csv')

test_label.head()
# train image names along with paths

test_image_name = [path+'test/'+str(each)+'.jpg' for each in test_label['image_id'].values.tolist()]

test_image_name[:5]
# preparing data by processing images using opencv

ROWS = 64

COLS = 64

CHANNELS = 1



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        if i%2000 == 0: print('Processed {} of {}'.format(i, count))    

    return data



train = prep_data(train_image_name)

test = prep_data(test_image_name)
optimizer = RMSprop(lr=1e-4)

objective = 'categorical_crossentropy'





def flower_recognition():

    

    model = Sequential()



    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, ROWS, COLS), activation='relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))







    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    

    model.add(Dense(103, activation='softmax'))



#     model.add(Activation('softmax'))



    model.compile(loss=objective, optimizer='adam', metrics=['accuracy'])

    return model





model = flower_recognition()
model.summary()
labs = to_categorical(train_label['category'])
nb_epoch = 10

batch_size = 16



## Callback for loss logging per epoch

class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))



early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        

        

history = LossHistory()
model.fit(train, labs, batch_size=batch_size, epochs=nb_epoch,

              validation_split=0.25, verbose=1, shuffle=True, callbacks=[history, early_stopping])
predictions = model.predict(test)
max(predictions[0])
loss = history.losses

val_loss = history.val_losses



plt.figure(figsize=(33, 15))

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('VGG-16 Loss Trend')

plt.plot(loss, 'blue', label='Training Loss')

plt.plot(val_loss, 'green', label='Validation Loss')

plt.xticks(range(0, nb_epoch)[0::2])

plt.legend()

plt.show()
predicted = []

for i in range(len(predictions)):

    predicted.append(np.argmax(predictions[i]))
test_label['category'] = predicted

test_label.head()
test_label.to_csv('test_submission.csv', index=False)