# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

    break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
directory = '/kaggle/input/rockpaperscissors'

print(os.listdir(directory))
labels = ['paper','scissors','rock']

nb = len(labels)
import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
def input_target_split(train_dir,labels):

    dataset = []

    count = 0

    for label in labels:

        folder = os.path.join(train_dir,label)

        for image in os.listdir(folder):

            img=load_img(os.path.join(folder,image), target_size=(150,150))

            img=img_to_array(img)

            img=img/255.0

            dataset.append((img,count))

        print(f'\rCompleted: {label}',end='')

        count+=1

    random.shuffle(dataset)

    X, y = zip(*dataset)

    

    return np.array(X),np.array(y)
X, y = input_target_split(directory,labels)
import matplotlib.pyplot as plt
plt.figure(figsize = (15 , 9))

n = 0

for i in range(15):

    n+=1

    plt.subplot(5 , 5, n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.3)

    plt.imshow(X[i])

    plt.title(f'Label: {labels[y[i]]}')
np.unique(y,return_counts=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

print(np.unique(y_train,return_counts=True),np.unique(y_test,return_counts=True))
datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             rotation_range=20,

                             zoom_range=0.2,

                             width_shift_range = 0.2,

                             height_shift_range = 0.2,

                             shear_range=0.1,

                             fill_mode="nearest")



testgen = ImageDataGenerator()



datagen.fit(X_train)

testgen.fit(X_test)
y_train = np.eye(nb)[y_train]

y_test = np.eye(nb)[y_test]
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(150,150,3), activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(units=512, activation='relu'))

model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
filepath= "model_cnn_final.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)



early_stopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)



# learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 

#                                             patience=3, 

#                                             verbose=1, 

#                                             factor=0.2, 

#                                             min_lr=0.00001)



callbacks_list = [

        checkpoint,

        early_stopping,

#         learning_rate_reduction

    ]

hist = model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),

                                        validation_data=testgen.flow(X_test,y_test,batch_size=32),

                                        epochs=50,

                                        callbacks=callbacks_list)
model_saved = tf.keras.models.load_model('model_cnn_final.h5')
y_pred = model_saved.predict(X_test)

pred = np.argmax(y_pred,axis=1)

print(pred)
ground = np.argmax(y_test,axis=1)
from sklearn.metrics import classification_report



print(classification_report(ground,pred,target_names = labels))
test = pd.DataFrame()
test.to_csv("sample.csv")
tf.__version__