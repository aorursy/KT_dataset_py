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
import matplotlib.pyplot as plt

import seaborn as sns

import os

import cv2

import random



%matplotlib inline

plt.style.use('ggplot')
train_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train"

val_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val"

test_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test"
labels = ['PNEUMONIA', 'NORMAL']

img_size = 220

training_data = []

val_data = []

test_data = []





def get_data(data_dir, flag=None):

    for label in labels:

        

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        for img in os.listdir(path):

            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img_arr, (img_size, img_size))

                if flag == 0:

                    training_data.append([img, class_num])

                elif flag == 1:

                    val_data.append([img, class_num])

                else:

                    test_data.append([img, class_num])

            except Exception as e:

                print(e)

                
get_data(train_dir, flag=0)

get_data(val_dir, flag=1)

get_data(test_dir)
fig = plt.figure(figsize=(10,6))

(ax1, ax2) = fig.subplots(1,2)



ax1.imshow(training_data[0][0])

ax1.set_title(labels[training_data[0][1]])

ax1.axis('off')



ax2.imshow(training_data[-1][0])

ax2.set_title(labels[training_data[-1][1]])

ax2.axis('off')

plt.show()

def get_features_and_labels(data):

    random.shuffle(data)

    X  = []

    y = []

    for feature, label in data:

        X.append(feature)

        y.append(label)

    

    return X, y



X_train, y_train = get_features_and_labels(training_data)

X_val, y_val = get_features_and_labels(val_data)

X_test, y_test = get_features_and_labels(test_data)
sns.countplot(y_train)
def preprocess_data(X, y):

    X = np.array(X)/255    #normalize data

    X = X.reshape(-1, img_size, img_size, 1)   #reshape data

    y = np.array(y)     #convert to numpy array

    

    return X, y





X_train, y_train = preprocess_data(X_train, y_train)

X_val, y_val = preprocess_data(X_val, y_val)

x_test, y_test = preprocess_data(X_test, y_test)
#imports



import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=40,

    zoom_range=0.2,

    width_shift_range=0.4,

    height_shift_range=0.4,

    fill_mode='nearest'

)



datagen.fit(X_train)
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout

from tensorflow.keras.optimizers import Adam
def build_model():

    model = Sequential()

    

    model.add(Conv2D(32, (3,3), strides = 1, padding='same', activation='relu', input_shape=(220, 220, 1), name='conv1' ))

    model.add(MaxPooling2D((2,2), strides = 2, padding='same', name='pool1'))

    

    model.add(Conv2D(64, (3,3), strides = 1, padding='same', activation='relu', name='conv2' ))

    model.add(MaxPooling2D((2,2), strides = 2, padding='same', name='pool2'))

    

    model.add(Conv2D(128, (3,3), strides = 1, padding='same', activation='relu', name='conv3' ))

#     model.add(BatchNormalization(name='bn3'))

    model.add(MaxPooling2D((2,2), strides = 2, padding='same', name='pool3'))

    

    model.add(Conv2D(256, (3,3), strides = 1, padding='same', activation='relu', name='conv4' ))

    model.add(MaxPooling2D((2,2), strides = 2, padding='same', name='pool4'))

    

    model.add(Conv2D(512, (3,3), strides = 1, padding='same', activation='relu', name='conv5' ))

    model.add(BatchNormalization(name='bn5'))

    model.add(MaxPooling2D((2,2), strides = 2, padding='same', name='pool5'))

    

    model.add(Flatten())

    model.add(Dense(1024, activation='relu', name='fc1'))

    model.add(Dropout(0.7, name='Dropout1'))

    model.add(Dense(512, activation='relu', name='fc2'))

    model.add(Dropout(0.2, name="Dropout2"))

    model.add(Dense(1, activation='sigmoid', name='fc3'))

    

    opt = Adam(lr = 0.0001, decay=1e-5)

    

    

    model.compile(optimizer=opt, 

                 loss='binary_crossentropy',

                 metrics=['acc'])

    

    return model
model = build_model()

model.summary()
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),

                    steps_per_epoch = len(X_train) / 32,

                    epochs=20, 

                    validation_data = datagen.flow(X_val, y_val, batch_size=32)

                   )
epochs = [i for i in range(20)]

fig = plt.figure(figsize=(20, 6))

(ax1, ax2) = fig.subplots(1,2)



ax1.plot(epochs, history.history['acc'], color='r')

ax1.plot(epochs, history.history['val_acc'], color='b')

ax1.set_xticks(epochs)

ax1.set_title('Accuracy')

ax1.set_xlabel('Epochs')

ax1.set_ylabel('Training & val Accuracy')

ax1.legend()





ax2.plot(epochs, history.history['loss'], color='r')

ax2.plot(epochs, history.history['val_loss'], color='b')

ax2.set_title('loss')

ax2.set_xticks(epochs)

ax2.set_xlabel('Epochs')

ax2.set_ylabel('Training & val loss')

ax2.legend()





plt.show()
predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1, -1)[0]

predictions[:20]
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions, target_names = ['Pneumonia (0)', 'Normal (0)']))
conf_matrix = confusion_matrix(y_test, predictions)

conf_matrix
sns.heatmap(conf_matrix, annot=True, fmt=".0f")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
correct_pred = np.nonzero(predictions == y_test)[0]

incorrect_pred = np.nonzero(predictions != y_test)[0]
plt.figure(figsize=(20,8))

i = 0

for c in correct_pred[:8]:

    plt.subplot(2, 4, i+1)

    plt.imshow(x_test[c].reshape(220, 220), interpolation=None)

    plt.title("Actual {}, Predicted {}".format(y_test[c], predictions[c]))

    plt.xticks([])

    plt.yticks([])

    i+=1
plt.figure(figsize=(20,8))

i = 0

for c in incorrect_pred[:8]:

    plt.subplot(2, 4, i+1)

    plt.imshow(x_test[c].reshape(220, 220), interpolation=None, cmap='gray')

    plt.title("Actual {}, Predicted {}".format(y_test[c], predictions[c]))

    plt.xticks([])

    plt.yticks([])

    i+=1