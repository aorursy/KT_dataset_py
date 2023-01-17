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

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam



%matplotlib inline

plt.style.use('ggplot')
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')   #Loads training data

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')     #Loads testing data
#Lets Get our features and labels



y_train = train_df['label']

X_train = train_df.drop('label', axis=1)
#preprocess our data



def preprocess_data(data):

    processed_data = data.values.reshape(-1, 28,28, 1)

    processed_data = processed_data / 255.0

    

    return processed_data



X_train = preprocess_data(X_train)

test = preprocess_data(test_df)
#make our labels

y_train = to_categorical(y_train, num_classes=10)
fig = plt.figure(figsize=(10,10))

ax = fig.subplots(5, 5)



count = 0

for i in range(5):

    for j in range(5):

        ax[i,j].imshow(X_train[count].reshape(28, 28), cmap='gray')

        count+=1       
plt.figure(figsize=(8,6))

sns.countplot(train_df.label.values)

plt.show()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
datagen = ImageDataGenerator(

    horizontal_flip=False,

    vertical_flip=False,

    rotation_range = 20, 

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

#     fill_mode='nearset'

)



datagen.fit(X_train)
def create_model():

    model = tf.keras.models.Sequential()

    

    model.add(Conv2D(16, (3,3), strides=1, padding='same', activation='relu', input_shape=(28, 28,1), name='conv1'))

    model.add(MaxPooling2D((2,2),strides=2, padding='same' ,name='pool1'))

    

    model.add(Conv2D(32, (3,3), strides=1, padding='same', activation='relu',name='conv2'))

    model.add(MaxPooling2D((2,2),strides=2, padding='same' ,name='pool2'))

    

    model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu',name='conv3'))

    model.add(MaxPooling2D((2,2),strides=2, padding='same' ,name='pool3'))

    

    model.add(Conv2D(128, (3,3), strides=1, padding='same', activation='relu',name='conv4'))

    model.add(MaxPooling2D((2,2),strides=2, padding='same' ,name='pool4'))

    

    model.add(Conv2D(264, (3,3), strides=1, padding='same', activation='relu',name='conv5'))

    model.add(MaxPooling2D((2,2),strides=2, padding='same' ,name='pool5'))

    

    model.add(Flatten())

    model.add(Dense(1024, name='Dense1', activation='relu'))

    model.add(Dense(512, name='Dense2', activation='relu'))

    model.add(Dense(10, name='Dense3', activation='softmax'))

    



    

    model.compile(optimizer='adam',

                 loss='categorical_crossentropy',

                 metrics=['acc'])

    return model
model = create_model()

model.summary()
batch_size=128

steps_per_epoch = X_train.shape[0] // batch_size

epochs=30



history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),

                               validation_data = (X_val, y_val),

                               epochs=epochs,

                                steps_per_epoch = steps_per_epoch

                             )
epoch = [i for i in range(30)]

fig = plt.figure(figsize=(20, 8))

(ax1, ax2) = fig.subplots(1,2)



ax1.plot(epoch, history.history['acc'], color='r')

ax1.plot(epoch, history.history['val_acc'], color='b')

ax1.set_title("Train and Validation Accuracy")

ax1.set_xticks(epoch)

ax1.set_xlabel('Epochs')

ax1.set_ylabel('Train and val Accuracy')

ax1.legend()





ax2.plot(epoch, history.history['loss'], color='r')

ax2.plot(epoch, history.history['val_loss'], color='b')

ax2.set_title("Train and Validation loss")

ax2.set_xticks(epoch)

ax2.set_xlabel('Epochs')

ax2.set_ylabel('Train and val Loss')

ax2.legend()



plt.show()
val_predicted = model.predict_classes(X_val)



new_y_val_labels =  [] 

[new_y_val_labels.append(np.argmax(i)) for i in y_val ]



conf_matrix = confusion_matrix(new_y_val_labels, val_predicted )

plt.figure(figsize=(8,6))

sns.heatmap(conf_matrix, annot=True, fmt='.0f')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
predicts = model.predict(test)

predicts= np.argmax(predicts,axis = 1)

predicts = pd.Series(predicts,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predicts],axis = 1)

submission.to_csv("submission.csv",index=False)