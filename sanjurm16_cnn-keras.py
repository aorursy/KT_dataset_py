# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
X = train.drop(['label'], axis = 1)
X = X / 255
X = X.values.reshape(-1,28,28,1)
plt.imshow(X[0][:,:,0])
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten

#create model

model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))


from keras.utils import to_categorical

#one-hot encode target column

y = train['label']

y_train = to_categorical(y)
y_train[0]
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y_train, random_state=4)
test_X[0]
#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model

model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=3)
test = pd.read_csv('../input/test.csv')

test = test / 255

test = test.values.reshape(-1,28,28,1)
test_pred = model.predict(test)
submission = pd.DataFrame()

submission['ImageId'] = range(1, (len(test)+1))

submission['Label'] = np.argmax(test_pred, axis=1)
submission.head()
submission.shape
submission.to_csv("submission.csv", index=False)