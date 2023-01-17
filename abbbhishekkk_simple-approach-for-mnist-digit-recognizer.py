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
import matplotlib.pyplot as plt

import seaborn as sns

import keras

%matplotlib inline
from keras.layers import Conv2D,pooling,Flatten

from keras.layers.pooling import MaxPool2D
import pandas as pd

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
mnist_test
mnist_train.shape
from collections import Counter
Counter(mnist_train['label'])
sns.countplot(mnist_train['label'])
x_train=(mnist_train.iloc[:,1:].values).astype('float32')

y_train=(mnist_train.iloc[:,0].values).astype('int')

y_train=keras.utils.to_categorical(y_train)
x_train=x_train/255.0

Mnist_Test=(mnist_test.values).astype('float32')/255.0
mnist_train.head()
import sklearn



from sklearn.model_selection import train_test_split

x_train=x_train.reshape(-1,28,28,1)
X_train,X_test,Y_train,y_test=train_test_split(x_train,y_train)
from keras.models import Sequential

import keras

from keras.layers import Dense

from keras.callbacks.callbacks import EarlyStopping
Modelnew=Sequential()

b=EarlyStopping(patience=3,monitor='val_loss')

from keras.layers import Dropout
Modelnew.add(Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=(28,28,1)))

Modelnew.add(Conv2D(filters=32,kernel_size=3,activation='relu'))

Modelnew.add(Dropout(0.2))

Modelnew.add(Flatten())

Modelnew.add(Dense(10,activation='softmax'))
Modelnew.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

Modelnew.fit(X_train,Y_train,validation_data=(X_test,y_test),epochs=15,batch_size=32,callbacks=[b])
Modelnew.summary()
Modelnew.evaluate(X_test,y_test)
pred=Modelnew.predict(X_test)
Y_pred_classes = np.argmax(pred, axis = 1)

Y_Act_Classes=np.argmax(y_test,axis=1)
from keras.models import Sequential

import keras

from keras.layers import Dense

from keras.callbacks.callbacks import EarlyStopping
model=Sequential()

a=EarlyStopping(patience=3,monitor='accuracy')
model.add(Dense(100,activation='relu',input_shape=(784,)))

model.add(Dense(100,activation='relu'))

model.add(Dense(100,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
y_train.shape
model.fit(X_train,Y_train,epochs=30,callbacks=[a],batch_size=32)
model.summary()
model.evaluate(X_test,y_test)
pred=model.predict(X_test)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
Y_pred_classes = np.argmax(pred, axis = 1)

Y_Act_Classes=np.argmax(y_test,axis=1)
Y_Act_Classes,Y_pred_classes
from sklearn.metrics import confusion_matrix,auc,f1_score,classification_report
confusion_matrix(Y_Act_Classes,Y_pred_classes)
print(classification_report(Y_Act_Classes,Y_pred_classes))

test_data=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_data
Modelnew.history.history
test_data=test_data.values.reshape(-1,28,28,1)
predicted_classes = Modelnew.predict_classes(test_data)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})

submissions.to_csv("new.csv", index=False, header=True)
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")