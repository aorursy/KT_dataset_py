# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Input, Dropout

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
X = train[train.columns[1:]].values

y = train['label']


X = X / 255.

y = np_utils.to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)




model = Sequential()

model.add(Dense(X_train.shape[1], input_shape=(784,), activation='relu'))

model.add(Dropout(.5))

model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100)


test = pd.read_csv('../input/test.csv')

test = test / 255.
pred = model.predict_classes(test.values)
test['Label'] = pred

test['ImageId'] = range(1,test.shape[0] + 1)
test[['ImageId', 'Label']].to_csv('submission.csv', index=False)