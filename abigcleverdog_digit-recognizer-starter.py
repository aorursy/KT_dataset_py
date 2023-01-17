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

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.utils.np_utils import to_categorical



from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

  rotation_range=10,

  zoom_range=0.1,

  width_shift_range=0.1,

  height_shift_range=0.1

)
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

ssub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train.head(2)
ssub.head()
y = train.label

X = train.drop('label', axis=1)
y.value_counts()
X = X/255.0

test = test/255.0
X = X.to_numpy().reshape(-1, 28,28,1)

test = test.to_numpy().reshape(-1, 28,28,1)
plt.imshow(X[0][:,:,0], cmap='gray');
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2)

i = 1

for sub_train, sub_test in kfold.split(X, y):

    yy = to_categorical(y.copy(), num_classes = 10)

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X.shape[1:]))

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(rate=0.5))

    model.add(Dense(10, activation='softmax'))

    

    model.compile(

        loss='categorical_crossentropy', 

        optimizer='adam', 

        metrics=['accuracy']

    )

    

    model.fit_generator(datagen.flow(X[sub_train], yy[sub_train], batch_size=50), epochs=1, verbose=1)

    

    y_pred = model.predict(X[sub_test])

    

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    # Convert validation observations to one hot vectors

    Y_true = np.argmax(yy[sub_test],axis = 1) 

    # compute the confusion matrix

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

#     cm = confusion_matrix(yy[test], y_pred)



    print(i,'*'*10)

    print(confusion_mtx)

    plt.imshow(confusion_mtx, interpolation='nearest')

    i += 1

    

yy = to_categorical(y.copy(), num_classes = 10)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X.shape[1:]))

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(10, activation='softmax'))



model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy']

)



model.fit_generator(datagen.flow(X, yy, batch_size=100), epochs=5, verbose=1)



y_pred = model.predict(test)



y_submission = np.argmax(y_pred,axis = 1)
submission = pd.DataFrame({'ImageId': range(1,28001), 'Label':y_submission})

submission.head()
submission.to_csv('cnn_submission_01.csv',index=False)