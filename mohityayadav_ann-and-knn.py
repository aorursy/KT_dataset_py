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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



np.random.seed(2)

import keras

from sklearn.preprocessing import MinMaxScaler





from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils import np_utils

from sklearn import metrics



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix





from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential



from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator



# Load the data

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()

test.head()
print(train.shape)

print(test.shape)
train_images= train.drop(labels = ["label"],axis = 1)

train_labels=train["label"]
print(train_images.shape)

print(train_labels.shape)
sns.countplot(train_labels)
#Train Test Split

X_train, X_test, y_train, y_test = train_test_split( train_images, train_labels, test_size=0.2, random_state=42)
#check for the null values

print(X_train.isnull().sum().max())

print(X_test.isnull().sum().max())
#Normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)
#Model

import keras



model=keras.models.Sequential()

model.add(Dense(units = 128, kernel_initializer = 'he_uniform',activation='relu',input_dim = 784))

model.add(Dense(10,activation='softmax'))

model.compile(loss="categorical_crossentropy",

              optimizer="sgd",

              metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=100)
#prediction

predicted = model.predict(X_test)

y_head = predicted.argmax(axis=1).reshape(-1,1)
from sklearn import metrics



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#Evaluation

print(accuracy_score(y_test.argmax(axis=1), y_head))
test = test.values
test.shape
# Making Predictions on Test Data

y_head_test = model.predict(test)

result_test = y_head_test.argmax(axis=1)

#Visualising predictions

for i in range(1,5):

    index = np.random.randint(1,28001)

    plt.subplot(3,3,i)

    plt.imshow(test[index].reshape(28,28))

    plt.title("Predicted Label : {} ".format(result_test[index]))

plt.subplots_adjust(hspace = 1.2, wspace = 1.2)

plt.show()
import time

from sklearn import metrics



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors = 1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_test,y_pred))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

import time

from sklearn import metrics



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors = 3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_test,y_pred))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

import time

from sklearn import metrics



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors = 5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_test,y_pred))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))
k = 1



model = KNeighborsClassifier(n_neighbors = k)

model.fit(X_train, y_train)

y_pred = model.predict(test)

result_test = y_pred.argmax(axis=1)

#Visualising predictions

for i in range(1,5):

    index = np.random.randint(1,28001)

    plt.subplot(3,3,i)

    plt.imshow(test[index].reshape(28,28))

    plt.title("Predicted Label : {} ".format(result_test[index]))

plt.subplots_adjust(hspace = 1.2, wspace = 1.2)

plt.show()
result_test = y_pred.argmax(axis=1)

#Visualising predictions

for i in range(1,5):

    index = np.random.randint(1,28001)

    plt.subplot(3,3,i)

    plt.imshow(test[index].reshape(28,28))

    plt.title("Predicted Label : {} ".format(result_test[index]))

plt.subplots_adjust(hspace = 1.2, wspace = 1.2)

plt.show()
hello