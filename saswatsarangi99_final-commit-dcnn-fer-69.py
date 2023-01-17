import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image

from tensorflow.keras.layers import Conv2D,Dense,AveragePooling2D,MaxPooling2D, Flatten, Dropout 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))
data = pd.read_csv('/kaggle/input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')
data.shape
data.head()
data.Usage.value_counts()
emotion_map = {0: ' Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
emotion_counts

plt.figure(figsize=(6, 4))
plt.bar(emotion_counts.emotion, emotion_counts.number)
plt.title('Class distribution')
plt.xlabel('Emotions', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.show()

x = []
y = []
first = True
for line in open("/kaggle/input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv"):
    if first:
        first = False
    else:
        row = line.split(',')
        x.append([int(p) for p in row[1].split()])
        y.append(int(row[0]))
x, y = np.array(x) / 255.0, np.array(y)
x = x.reshape(-1, 48, 48, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
num_class = len(set(y))
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

datagen = ImageDataGenerator(
    zoom_range = 0.15,
    height_shift_range = 0.15,
    width_shift_range = 0.15,
    rotation_range = 10,
    horizontal_flip = True,
    vertical_flip = True
)
datagen.fit(x_train)

cnn_model = Sequential()

Input_shape = (48, 48, 1)

cnn_model = Sequential()


cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal', padding='same',input_shape = Input_shape))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.3))


cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))

cnn_model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.5))


cnn_model.add(Flatten())

cnn_model.add(Dense(units = 1024, activation = 'relu', kernel_initializer='he_uniform'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(units = 512, activation = 'relu', kernel_initializer='he_uniform'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.3))

cnn_model.add(Dense(units = 256, activation = 'relu', kernel_initializer='he_uniform'))
cnn_model.add(BatchNormalization())

cnn_model.add(Dense(units = 7, activation = 'softmax'))

cnn_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999), metrics = ['accuracy'])
cnn_model.summary()
history = cnn_model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=350, validation_data=(x_test, y_test), verbose=1)
score = cnn_model.evaluate(x_test, y_test)
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.show()