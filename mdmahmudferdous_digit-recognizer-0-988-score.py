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

%matplotlib inline



import tensorflow as tf

tf.__version__

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D, Dropout

from sklearn.metrics import classification_report, confusion_matrix
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sample_submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
plt.figure(figsize=(10,5))

sns.countplot(train.label)
X_train=train.drop(columns=['label']).values

y_train=train.label.values

#Normalize the data

X_train=tf.keras.utils.normalize(X_train, axis=1)

X_test=tf.keras.utils.normalize(test, axis=1).values
print(X_train.shape, y_train.shape, X_test.shape)
X_test1 = X_test.reshape(X_test.shape[0],28,28,1)

X_train1 = X_train.reshape(X_train.shape[0],28,28,1)
model=Sequential()

model.add(Conv2D(128, (3,3), input_shape=X_train1.shape[1:],strides=2))

model.add(Activation('relu'))



model.add(Conv2D(128, (3,3),strides=2))

model.add(Activation('relu'))





model.add(Conv2D(128, (3,3),strides=2))

model.add(Activation('relu'))



model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))



model.add(Dense(10))

model.add(Activation('softmax'))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

                metrics=['accuracy']

              )

model.fit(X_train1, y_train, epochs=50, validation_split=0.2) #Please change epochs equal to at least 5 while re-training

y_pred_train=model.predict_classes(X_train1)

y_pred=model.predict_classes(X_test1)
cm=confusion_matrix(y_train,y_pred_train)

cm=pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])

plt.figure(figsize=(10,10))

sns.heatmap(cm, cmap='Blues',linecolor='black',linewidths=1,annot=True,fmt='')
sample_submission.head()

submission=pd.DataFrame({'ImageId': sample_submission.ImageId,'Label':y_pred})

submission.to_csv('/kaggle/working/submission.csv',index=False)

check=pd.read_csv('/kaggle/working/submission.csv')

check.head()
X_test_1=X_test.reshape(X_test.shape[0],28,28)

plt.imshow(X_test_1[100])

plt.show()

print('Prediction: ', y_pred[100])