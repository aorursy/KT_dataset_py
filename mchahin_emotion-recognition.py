import matplotlib.pyplot as plt

import json

import os

import cv2

import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from skimage.filters import unsharp_mask

from keras.utils import to_categorical

from keras.layers import Activation, Convolution2D, Dropout, Conv2D

from keras.layers import AveragePooling2D, BatchNormalization

from keras.layers import GlobalAveragePooling2D

from keras.models import Sequential

from keras.layers import Flatten

from keras.models import Model

from keras.layers import Input

from keras.layers import MaxPooling2D

from keras.layers import SeparableConv2D

from keras.layers import Dense

from keras import layers

from keras.regularizers import l2

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
def get_emotion(emotion_id):

    """

    Maps emotion ids into emotion strings

    :param emotion_id:

    :return:

    """

    if emotion_id == 0:

        return 'Anger'

    elif emotion_id == 1:

        return 'Disgust'

    elif emotion_id == 2:

        return 'Fear'

    elif emotion_id == 3:

        return 'Happiness'

    elif emotion_id == 4:

        return 'Sadness'

    elif emotion_id == 5:

        return 'Surprised'

    elif emotion_id == 6:

        return 'Neutral'
os.listdir("../input")
frame = pd.read_csv('../input/fer2013.csv')
data = frame.iloc[:,:2]
data.head(10)
for i in range(7):

    print('{}: {}'.format(get_emotion(i),len(frame.where(frame['emotion']==i).dropna())))
anger = data.where(data['emotion']==0).dropna()

anger = anger.append(anger.head(3000))

len(anger)
disgust = data.where(data['emotion']==1).dropna()

disgust = disgust.append(disgust)

disgust = disgust.append(disgust)

disgust = disgust.append(disgust)

disgust = disgust.append(disgust)

len(disgust)
fear = data.where(data['emotion']==2).dropna()

fear = fear.append(fear.head(3000))

len(fear)
happiness = data.where(data['emotion']==3).dropna()

len(happiness)

# happiness = happiness.head(1000)
sadness = data.where(data['emotion']==4).dropna()

sadness = sadness.append(sadness.head(2000))

len(sadness)
surprised = data.where(data['emotion']==5).dropna()

surprised = surprised.append(surprised)

len(surprised)
neutral = data.where(data['emotion']==6).dropna()

neutral = neutral.append(neutral.head(2000))

len(neutral)
data = anger

data = data.append(disgust, ignore_index = True)

data = data.append(fear, ignore_index = True)

data = data.append(happiness, ignore_index = True)

data = data.append(sadness, ignore_index = True)

data = data.append(surprised, ignore_index = True)

data = data.append(neutral, ignore_index = True)
len(data)
X = np.asarray([np.fromstring(

    frame['pixels'][i], 

    sep=' ', 

).reshape(48, 48, 1) for i in range(len(frame))])



# standardize the values

X -= np.mean(X, axis=0)

X /= np.std(X, axis=0)



y = np.asarray(

    [int(frame['emotion'][i]) for i in range(len(frame))]

)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42

)

X_train.shape
import matplotlib.pyplot as plt

import keras



from IPython.display import clear_output



%matplotlib inline



class PlotLearning(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        self.max_val_acc = 0

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        if self.max_val_acc< logs.get('val_acc'):

            self.max_val_acc = logs.get('val_acc')

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)



        clear_output(wait=True)



        ax1.set_yscale('log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()



        ax2.plot(self.x, self.acc, label="accuracy")

        ax2.plot(self.x, self.val_acc, label="validation accuracy,(max: {})"

                 .format(round(self.max_val_acc,2)))

        ax2.legend()

        print("Max validation accuracy {}".format(self.max_val_acc))

        plt.show();





plot = PlotLearning()
num_features = 64

num_labels = 7

batch_size = 64

epochs = 50

width, height = 48, 48



model = Sequential()



model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(2*2*2*num_features, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(2*2*num_features, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(2*num_features, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_labels, activation='softmax'))
model.compile(

    optimizer='adam', 

    loss='categorical_crossentropy',

    metrics=['accuracy']

)
model.fit(

    X_train, 

    y_train, 

    validation_data=(X_test, y_test), 

    epochs=epochs,

    callbacks=[plot],

    verbose=2

)
correct, samples = 0, 1000

for index in range(samples):

    a = get_emotion(model.predict(np.asarray([X_test[index]]))[0].argmax())

    b = get_emotion(y_test[index].argmax())

    if a==b:

        correct += 1
print('{0:.2f}%'.format((correct/samples)*100))
model.save_weights('er_weights.h5')



with open('er_arch.json', 'w') as f:

    f.write(model.to_json())



print('Model saved.')