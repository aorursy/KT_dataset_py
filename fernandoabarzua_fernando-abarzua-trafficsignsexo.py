import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AvgPool2D, BatchNormalization, Reshape
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
# Lecture du répertoire Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pickle
with open('../input/traffic-signs-preprocessed/data3.pickle', 'rb') as f:

    data = pickle.load(f, encoding='latin1')
X_train = data['x_train']

y_train = data['y_train']
X_train.shape
print(X_train)
X_train = X_train.transpose(0,2,3,1)

print(X_train)
X_train.shape
y_train.shape
y_train[0]
labels = ['Speed limit (20km/h)',

 'Speed limit (30km/h)',

 'Speed limit (50km/h)',

 'Speed limit (60km/h)',

 'Speed limit (70km/h)',

 'Speed limit (80km/h)',

 'End of speed limit (80km/h)',

 'Speed limit (100km/h)',

 'Speed limit (120km/h)',

 'No passing',

 'No passing for vehicles over 3.5 metric tons',

 'Right-of-way at the next intersection',

 'Priority road',

 'Yield',

 'Stop',

 'No vehicles',

 'Vehicles over 3.5 metric tons prohibited',

 'No entry',

 'General caution',

 'Dangerous curve to the left',

 'Dangerous curve to the right',

 'Double curve',

 'Bumpy road',

 'Slippery road',

 'Road narrows on the right',

 'Road work',

 'Traffic signals',

 'Pedestrians',

 'Children crossing',

 'Bicycles crossing',

 'Beware of ice/snow',

 'Wild animals crossing',

 'End of all speed and passing limits',

 'Turn right ahead',

 'Turn left ahead',

 'Ahead only',

 'Go straight or right',

 'Go straight or left',

 'Keep right',

 'Keep left',

 'Roundabout mandatory',

 'End of no passing',

 'End of no passing by vehicles over 3.5 metric tons']



plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(X_train[i])

    plt.title(labels[y_train[i]])
X_test = data['x_test'].transpose(0,2,3,1)

y_test = data['y_test']

num_classes = len(labels)
# Normalisation

X_train = X_train / 255

X_test = X_test / 255
y_train1 = to_categorical(y_train)

y_test1 = to_categorical(y_test)
y_test1
# Réseau convolutionnel simple

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))



# Compilation du modèle

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage

train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=200, verbose=1)
# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
print(train.history['accuracy'])
print(train.history['val_accuracy'])
def plot_scores(train) :

    accuracy = train.history['accuracy']

    val_accuracy = train.history['val_accuracy']

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')

    plt.plot(epochs, val_accuracy, 'r', label='Score validation')

    plt.title('Scores')

    plt.legend()

    plt.show()
plot_scores(train)
# Prediction

y_cnn = model.predict_classes(X_test)
cm = confusion_matrix(y_cnn,y_test)

print(cm)

plt.figure(figsize = (12,10))
plt.figure(figsize=(15,25))

n_test = X_test.shape[0]

i=1

for j in range(len(X_test)) :

    if (y_cnn[j] != y_test[j]) & (i<48):

        plt.subplot(12,4,i)

        plt.axis('off')

        plt.imshow(X_test[j])

        pred_classe = y_cnn[j].argmax(axis=-1)

        plt.title('%s / %s' % (labels[int(y_cnn[j])], labels[int(y_test[j])]))

        i+=1
# Modèle CNN plus profond

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=50, batch_size=200, verbose=1)
# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
from keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

vgg16.trainable = False
vgg16.summary()
model = Sequential()

model.add(vgg16)

model.add(Flatten())

model.add(Dense(num_classes, activation='relu'))

model.summary()


# Compilation du modèle

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=200, verbose=1)
scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))