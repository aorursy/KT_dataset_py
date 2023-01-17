# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils
train_df = pd.read_csv('../input/train.csv')
print(train_df.shape)
train_df[train_df.columns[1:]].head()
X_train = train_df[train_df.columns[1:]].as_matrix()
X_train.shape
X_train_reshaped = X_train / 255
Y_train = train_df[train_df.columns[0:1]]
Y_train.shape
y_train_reshaped = np_utils.to_categorical(Y_train, num_classes=10)
y_train_reshaped.shape
y_train_reshaped[5]
model = Sequential()
model.add(Dense(units=512, input_shape = (784, )))
model.add(Activation('relu'))
model.add(Dense(units = 512))
model.add(Activation('relu'))
model.add(Dense(units = 10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(batch_size=128,epochs=4,x=X_train_reshaped, y=y_train_reshaped, verbose=1)
predicted_classes_train = model.predict_classes(X_train_reshaped)
predicted_classes_train[1:10]
a = X_train_reshaped[:5]
a
y_train_reshaped[0:5]
#so the first 5 labels in order are: 1,0, 1, 4, 0
#understand predict prob vs predict classes
b = model.predict(a)
b
b[0]
sum(b[0])
c = model.predict_proba(a)

c
#c is same as b; so no diff between predict and predict prob
d = model.predict_classes(a)

d
#looks like the model looks up the classes and does the mapping automatically
test = pd.read_csv('../input/test.csv')
test.head()
test_reshaped = test.as_matrix() / 255
test_reshaped.shape
predictions_test_model1 = model.predict_classes(test_reshaped)
np.unique(predictions_test_model1)
predictions_test_model1[1:5]
e = enumerate(predictions_test_model1)
predicted_probs = [[index + 1, x] for index, x in enumerate(predictions_test_model1)]
save_file = np.array(predicted_probs)
save_file.shape
np.savetxt('Shenba_MNIST_25Sep.csv', save_file, delimiter=',')