# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/mnist-digit-recognizer"))



print(os.listdir("../input/digit-recognizer"))

# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 



import random as rn



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline





from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import train_test_split



from keras.utils.np_utils import to_categorical

from keras.utils import np_utils



from keras.models import Sequential

from keras.layers import  Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,GlobalAveragePooling2D



from keras.optimizers import Adadelta, RMSprop, Adam

from keras.losses import categorical_crossentropy

from keras.wrappers.scikit_learn import KerasClassifier



import tensorflow as tf





from keras.preprocessing.image import ImageDataGenerator
!ls ../input
train = pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")
X_train = train.drop(["label"],axis = 1)

Y_train = train["label"]

X_test=test
X_train = X_train/255.0

X_test = X_test/255.0
X_train = X_train.values.reshape(len(X_train), 28, 28,1)

X_test = X_test.values.reshape(len(X_test), 28, 28,1)
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

# Set the random seed

random_seed = 3

# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# building a linear stack of layers with the sequential model

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1))) # 26

model.add(BatchNormalization())



model.add(Conv2D(16,(3, 3), dilation_rate=(3, 3), activation='relu')) # 24

model.add(BatchNormalization())



model.add(Conv2D(21, (3, 3), activation='relu')) # 22

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2))) #11



model.add(Conv2D(18, (3, 3), activation='relu')) # 9

model.add(BatchNormalization())







model.add(Conv2D(27,(3, 3),dilation_rate=(3, 3), activation='relu'))#7

model.add(BatchNormalization())



model.add(Conv2D(10,(1, 1), activation='relu'))#7

model.add(BatchNormalization())







model.add(GlobalAveragePooling2D())





#model.add(Flatten())

model.add(Activation('softmax'))
model.summary()
# For classification problems we use sparse_categorical_crossentropy loss function 

model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
%%time

batch_size=27

history = model.fit(X_train, Y_train, epochs=40,verbose=1,validation_data = (X_val,Y_val),batch_size=batch_size)
val_loss,val_acc = model.evaluate(X_val, Y_val, verbose=0)

print(val_acc)
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])





plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

#plt.legend(['test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()
print("Validation Accuracy:",val_acc)
# predict results

results = model.predict(X_test)
# select the indix with the maximum probability

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_predictions.csv",index=False)