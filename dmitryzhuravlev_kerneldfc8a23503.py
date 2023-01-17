# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trainfile = "/kaggle/input/emotion-recognition-mlpo/data/train.csv"
testfile = "/kaggle/input/emotion-recognition-mlpo/data/test.csv"

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import cv2
from IPython.display import clear_output

from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense, Flatten
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, f1_score
from keras import backend as K

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] 
N_CLASSES = len(EMOTIONS)

img_width, img_height = (48,48)

def str_to_image(image_blob):
    image_string = image_blob[0].split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    return image_data
 
def csv_to_array(csv):
    X = csv.pixels.values
    X = np.apply_along_axis(str_to_image, 1, X[:,None])/255.
    X = X[...,None]
    return X

train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)

y_train = train.emotion.values
X_train = csv_to_array(train)
X_test= csv_to_array(test)

y_train = keras.utils.to_categorical(y_train, N_CLASSES)

N_IMAGES_TO_PLOT = 36
fig, axes = plt.subplots(nrows=np.sqrt(N_IMAGES_TO_PLOT).astype(int), 
                         ncols = np.sqrt(N_IMAGES_TO_PLOT).astype(int),
                         figsize=(10,10))

rand_indx = np.random.choice(np.arange(N_IMAGES_TO_PLOT), size=N_IMAGES_TO_PLOT, replace=False)
for ax, emotion_index, img in zip(axes.flatten(), train.emotion.values[rand_indx], X_train[rand_indx]):
    img = img[:,:,0] # [48,48,1] -> [48,48]
    ax.imshow(img, cmap='gray')
    emotion = EMOTIONS[emotion_index]
    ax.set_title(emotion)
plt.tight_layout()
plt.show()

train_count = train.groupby('emotion').count()
plt.bar(x = EMOTIONS,
        height=train_count.values.flatten())
plt.show()

if K.image_data_format() == 'channels_first': 
    input_shape = (1, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 1) 
X_train, X_test = X_train.reshape((28709, 48, 48, 1)), X_test.reshape((7178, 48, 48, 1))
input_size = X_train[0].shape
model = Sequential() 

model.add(Conv2D(24, (3, 3), padding='same', input_shape=input_size))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256)) 
model.add(Activation('relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(256)) 
model.add(Dropout(0.3)) 
model.add(Dense(N_CLASSES)) 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='nadam', 
              metrics=['accuracy'])

N_EPOCHS = 10
history_cnn = model.fit(X_train, 
                        y_train,
                        batch_size=100,
                        epochs=N_EPOCHS,
                        workers = 400)

plt.figure()
plt.plot(history_cnn.history['loss'], label='training loss')
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(history_cnn.history['accuracy'], '--s', color='r', label='training accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

def submit(model, X_test):
    prediction=model.predict(X_test,  workers=400)
    pred_classes = prediction.argmax(-1)
    df = pd.DataFrame(data = {'Id':np.arange(len(pred_classes)),
                              'Category':pred_classes})
    df.to_csv('submission.csv', index=False)

submit(model, X_test)