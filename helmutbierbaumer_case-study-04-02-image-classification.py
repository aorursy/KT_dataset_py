import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set()
import imageio
from PIL import Image
from random import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
#os.chdir('/kaggle/input/people-faces/Faces')
#os.chdir('/kaggle/input/people-faces-blurred-0x6/Faces_blurred_0x6')
#os.chdir('/kaggle/input/people-faces-blurred-0x8/Faces_blurred_0x8')
#os.chdir('/kaggle/input/people-faces-pixelated-15/Faces_pixelated_15')
os.chdir('/kaggle/input/people-faces-pixelated-10/Faces_pixelated_10')

images = os.listdir()
shuffle(images)
data = []
gender = []
for img in images:
    picture = imageio.imread(img)
    picture = cv2.resize(picture, (32, 32))
    if len(picture.shape) == 3:
        data.append(picture)
        male_female = img.split('_')[1]
        gender.append(int(male_female))
X = np.squeeze(data)
gender_labels = to_categorical(gender, num_classes=2)
(x_train, y_train), (x_test, y_test) = (X[:8500], gender_labels[:8500]), (X[8500:], gender_labels[8500:])
(x_valid, y_valid) = (x_test[:3500], y_test[:3500])
(x_test, y_test) = (x_test[3500:], y_test[3500:])
print("Total number of images:\t\t{:>6}".format(len(X)))
print("Number of training samples:\t{:>6}".format(len(y_train)))
print("Number of validation samples:\t{:>6}".format(len(y_valid)))
print("Number of test samples:\t\t{:>6}".format(len(y_test)))
len(y_train)+len(y_valid)+len(y_test) == len(X)
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=35, validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test)
print('Accurarcy Score: {}'.format(score[1]))
plt.figure(figsize=(15,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
g_labels = ['Male', 'Female']
y_pred = model.predict(x_test)

fig = plt.figure(figsize=(20,8))
for count, index in enumerate(np.random.choice(x_test.shape[0], size=12, replace=False)):
    ax = fig.add_subplot(3, 4, count+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[index]))
    correct = np.argmax(y_test[index])
    predicted = np.argmax(y_pred[index])
    color = 'green' if correct == predicted else 'red'
    ax.set_title('{} ({})'.format(g_labels[predicted], g_labels[correct]), color=color)
    
plt.show()
    