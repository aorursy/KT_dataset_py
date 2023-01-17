# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # graph

from matplotlib import pyplot as plt



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
y_train = train['label']

X_train = train.drop('label', axis=1)
sns.set(rc={'figure.figsize':(9,7)})

sns.countplot(y_train)
y_train.value_counts()
null_train = X_train.isnull().sum()
null_train[null_train > 0]
null_test = test.isnull().sum()
null_test[null_test > 0]
X_train.describe()
np.sqrt(X_train.shape[1])
# Normalize the data



X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

test = test.values.reshape(test.shape[0], 28, 28, 1)
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train[i][:,:,0], cmap=plt.get_cmap('gray'))

plt.show()
# Create label encoding for the categorical values



y_train = to_categorical(y_train, num_classes=10)
y_train
# Create model



model = Sequential()

model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.output_shape
model.add(Conv2D(32, 3, 3, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))          
model.output_shape
model.add(Conv2D(64, 3, 3, activation='relu'))
model.output_shape
model.add(Conv2D(64, 3, 3, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.output_shape
model.add(Flatten())
model.output_shape
model.add(Dense(128, activation='relu'))
model.output_shape
model.add(Dropout(0.25)) 
model.add(Dense(10, activation='softmax'))
model.output_shape
# Compile model

model.compile(loss='categorical_crossentropy',

              optimizer='RMSprop',

              metrics=['accuracy'])

 

# 9. Fit model on training data

model.fit(X_train, y_train, 

          batch_size=32, nb_epoch=10, verbose=1)
y_pred = model.predict_classes(test, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),

                         "Label": y_pred})

submissions.to_csv("mnist_submission.csv", index=False, header=True)