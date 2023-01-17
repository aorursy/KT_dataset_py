# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Lecture des données 



import pandas as pd

import numpy as np



train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



print("Train",train.shape)

print("Test",test.shape)
#Préparation des données 



    #Train



y_train = train["label"]

x_train = train.drop(labels = ["label"],axis = 1)



print("Train",x_train.shape)

print("Labels train",y_train.shape)
    #On split le data set : test size 10% et train size 90%



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)



print("x_train shape",x_train.shape)

print("x_test shape",x_test.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_test.shape)
import keras

from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D

from keras.models import Model

from sklearn.metrics import accuracy_score, confusion_matrix

from keras.utils import to_categorical



#Normalise les données



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train = x_train / 255.0

x_test = x_test / 255.0



#Reshape les données



x_train = x_train.values.reshape((-1, 28, 28, 1))

x_test = x_test.values.reshape((-1, 28, 28, 1))



print(x_train.shape)

print(x_test.shape)



# One hot labels.

# Cela signifie qu'une colonne sera créée pour chaque catégorie de sortie et une variable binaire est entrée pour chaque catégorie.



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

print(y_train.shape)

print(y_test.shape)



#Les deux premières lignes : 

#premier nombre est le nombre d'images (37 800 pour x_train et 42 000 pour x_test). 

#Deuxième et troisième nombre la forme de chaque image (28x28). 

#Le dernier nombre est 1, ce qui signifie que les images sont en niveaux de gris.
from keras.models import Model

from keras.models import Sequential

from keras.layers import Dense,Flatten

from keras.layers import Conv2D, MaxPooling2D
#model = Sequential()

#model.add (Conv2D (64, kernel_size = (5,5), activation = "relu", input_shape = (28,28,1)))

#model.add (Conv2D (32, kernel_size = (5,5), activation = 'relu'))

#model.add (Flatten ())

#model.add (Dense (10, activation = "softmax"))
model = Sequential()

model.add (Conv2D (64, kernel_size = (5,5), activation = "relu", input_shape = (28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add (Conv2D (32, kernel_size = (5,5), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add (Flatten ())

model.add (Dense (10, activation = "softmax"))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit (x_train, y_train, validation_data = (x_test, y_test), epochs = 10)
Y_pred=model.predict(x_test)



results = np.argmax(Y_pred,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)

submission.head()
sub=pd.DataFrame({

    "ImageId":(np.arange(x_test.shape[0])+1),

    "Label" : Y_pred})



sub.to_csv("submission.csv",index=False)
loss, acc = model.evaluate(x_train, y_train, verbose=0)

print("loss: {0:.4f},  accuracy: {1:.4f}".format(loss, acc))
Model=model.fit (x_train, y_train, validation_data = (x_test, y_test), epochs = 10)
import os

import matplotlib.pyplot as plt



figure = plt.figure()

plt.subplot(2,1,1)

plt.plot(Model.history['accuracy'])

plt.plot(Model.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)

plt.plot(Model.history['loss'])

plt.plot(Model.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

figure