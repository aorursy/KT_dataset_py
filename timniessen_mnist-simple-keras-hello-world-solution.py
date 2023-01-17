# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import models

from keras import layers

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_train.shape
df_test.shape
from sklearn.model_selection import StratifiedShuffleSplit 



split = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

for train_index, test_index in split.split(df_train, df_train['label']):

    strat_train_set = df_train.loc[train_index]

    strat_test_set = df_train.loc[test_index]

    
strat_train_set['label'].hist()

strat_test_set['label'].hist()

plt.show()
y_train = strat_train_set['label'].values

strat_train_set.drop('label', axis=1, inplace=True)

X_train = strat_train_set.values
y_test = strat_test_set['label'].values

strat_test_set.drop('label', axis=1, inplace=True)

X_test = strat_test_set.values
some_digit = X_train[9]

some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation='nearest')

plt.axis('off')

print(y_train[9])

plt.show()
model = models.Sequential() 

model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) 

model.add(layers.Dense(10, activation='softmax'))



model.compile(optimizer='rmsprop',

                loss='categorical_crossentropy',

                metrics=['accuracy'])
X_train = X_train.astype('float32') / 255 

X_test = X_test.astype('float32') / 255

X_final = df_test.values.astype('float32') / 255
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from keras.utils import to_categorical



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
model.fit(X_train, y_train, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(X_test, y_test)

print(test_acc)
pred = model.predict_classes(X_final)
pred
submission = pd.DataFrame({

        "ImageId": list(range(1,len(pred)+1)),

        "Label": pred

    })

print(submission.head())

submission.to_csv('submission.csv', index=False)