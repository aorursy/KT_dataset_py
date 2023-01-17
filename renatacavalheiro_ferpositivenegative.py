# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# reading the file



import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Activation, Dropout, Flatten



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt

from sklearn import preprocessing



# labeling the file with the characteristics



label_map = ['Negative', 'Positive']

names=['emotion','pixels','usage']

filename = '../input/newfer2013/fer2013NewData.csv'

fer2013 = pd.read_csv(filename, names=names, na_filter=False)

columns = fer2013.columns.tolist()

print("Colunas: ", columns)



# classifying the emotions

# labels: 0 - Anger, 1 - Disgust, 2 - Fear, 3 - Happy, 4 - Sad, 5 - Surprise, 6 - Neutral

# positive emotions: 3, 5 and 6

# negative emotions: 0, 1, 2, and 4



    

print(fer2013.info())

print(fer2013.head(10), "\n")

print(fer2013.tail(10), "\n")

print(fer2013.describe(), "\n")



# getting the data and checking the size of the file



def getData(filename):

    # images are 48x48

    # N = 35888

    Y = []

    X = []

    first = True

    for line in open(filename):

        if first:

            first = False

        else:

            row = line.split(',')

            Y.append(int(row[0]))

            X.append([int(p) for p in row[1].split()])



    X, Y = np.array(X) / 255.0, np.array(Y)

    return X, Y



X, Y = getData(filename)

print(Y)

num_class = len(set(Y))

print(label_map)

print("Número de classes: ", num_class, "\n")



# keras with tensorflow backend

N, D = X.shape

X = X.reshape(N, 48, 48, 1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)

y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)



from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization



def my_model():

    model = Sequential()

    input_shape = (48,48,1)

    model.add(Conv2D(64, (2, 2), input_shape=input_shape, activation='relu', padding='same'))

    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))

    model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))

    model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(128))

#     model.add(BatchNormalization())

    

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(2))

    model.add(Activation('softmax'))

    

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.summary()

    

    return model

model=my_model()

model.summary()
# path_model='model_filter.h5' # save model at this location after each epoch

# K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one



import keras.backend as K







model=my_model() # create the model

K.set_value(model.optimizer.lr,1e-3) # set the learning rate

# fit the model

h=model.fit(x=X_train,     

            y=y_train, 

            batch_size=128, 

            epochs=50, 

            verbose=1, 

            validation_data=(X_test,y_test),

            shuffle=True

#             callbacks=[

#                 ModelCheckpoint(filepath=path_model),

#             ]

            )
from skimage import io



def emotion_analysis(emotions):

    objects = ['Negativa', 'Positiva']

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.9)

    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)

    plt.xticks(y_pos, objects)

    plt.ylabel('porcentagem')

    plt.title('emoção')

    print(plt.show())



y_pred=model.predict(X_test)

print("Shape: ", y_test.shape)



objects = ['Negativa', 'Positiva']

y_pos = np.arange(len(objects))

print(y_pos)



# image similar to the ones in the dataset

img = image.load_img('../input/example/example1.jpg', color_mode = "grayscale", target_size=(48, 48))

show_img=image.load_img('../input/example/example1.jpg', color_mode = "grayscale", target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i        



# classifying the emotions

# labels: 0 - Anger, 1 - Disgust, 2 - Fear, 3 - Happy, 4 - Sad, 5 - Surprise, 6 - Neutral

# positive emotions: 3, 5 and 6

# negative emotions: 0, 1, 2, and 4

        

print('Expression Prediction:',objects[ind])
# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

import seaborn as sns

import pandas as pd



from sklearn.metrics import confusion_matrix



y_pred = np.argmax(y_pred, axis=1)

y_test = np.argmax(y_test, axis=1)



cm = confusion_matrix(y_test, y_pred)

print(cm)



plt.figure(figsize = (5,5))

sns.heatmap(cm, annot = True,cmap='PuRd',cbar=False,linewidth=2,fmt='d')

plt.title('CNN')

plt.ylabel('True class')

plt.xlabel('Prediction class')

plt.show()