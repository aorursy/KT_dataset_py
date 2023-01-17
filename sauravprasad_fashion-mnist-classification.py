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
#importing libraries

from keras.models import Sequential

from keras.layers import Dense,Activation, Conv2D, MaxPool2D, Flatten, Dropout

from keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

#loading dataset

test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
train.head()
#converting the dataframe into numpy array

train= train.values



#dividng the data set into features and target and standardising by dividing 255

x = train[:,1:].reshape(-1,28,28,1)/255.0 #feature

y = train[:,0].astype(np.int32) # target
outfit_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#data visulisation 



plt.figure(figsize=(15, 15))

for i in range(36):

    plt.subplot(6, 6, i + 1)

    plt.axis("off")

    plt.imshow(x[i].reshape((28,28)))

    target = y[i]

    plt.title(outfit_names[target])

plt.show()
#converting into caetgorical varaible

from keras.utils import to_categorical

y = to_categorical(y)
from keras.models import Sequential

from keras.layers import Dense,Activation, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
#Building a Model

model = Sequential()



#adding layer

model.add(Conv2D(input_shape = (28,28,1), filters = 64, kernel_size = (3,3)))

model.add(Activation("relu"))

model.add(MaxPool2D())



#adding layer

model.add(Conv2D(filters = 128, kernel_size = (3,3)))

model.add(Activation("relu"))

model.add(MaxPool2D())







#flatting and adding dense layer

model.add(Flatten())

model.add(Dense(units=512))

model.add(Activation ("relu"))

model.add(Dropout(.5))

model.add(Dense(units = 10 ))

model.add(Activation ("softmax"))



#compling model

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])





early_stop = EarlyStopping(monitor="val_loss", mode = "min", verbose=1, patience = 3)
model.fit(x,y, validation_split = 0.3, epochs = 30, batch_size=32,callbacks = [early_stop])
loss = pd.DataFrame(model.history.history)
loss
plt.figure(figsize=(10, 10))



plt.subplot(2, 2, 1)

plt.plot(loss['loss'], label='Loss')

plt.plot(loss['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(loss['accuracy'], label='Accuracy')

plt.plot(loss['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title('Train - Accuracy')
#converting the dataframe into numpy array

test= test.values



#dividng the data set into features and target and standardising by dividing 255

x = test[:,1:].reshape(-1,28,28,1)/255.0 #feature

y = test[:,0].astype(np.int32) # target

#making predictions

predictions = model.predict_classes(x)
#classification report

from sklearn.metrics import classification_report

print(classification_report(y, predictions,target_names = outfit_names))
#data visulisation with prediction and actual class label

plt.figure(figsize=(20, 20))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.axis("off")

    plt.imshow(x[i].reshape((28,28)))

    plt.title(f"Prediction Class = {predictions[i]} \n Original_Class= {y[i]}")

plt.show()
#data visulisation with prediction and actual class label

plt.figure(figsize=(18, 18))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.axis("off")

    plt.imshow(x[i].reshape((28,28)))

    plt.title(f"Prediction Class = {outfit_names[predictions[i]]} \n Original_Class= {outfit_names[y[i]]}")

plt.show()