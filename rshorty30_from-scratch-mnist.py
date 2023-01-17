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
train = pd.read_csv('../input/train.csv')

train.head()
train.info()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, pd.get_dummies(train['label']),

                                                                        test_size = 0.15,

                                                                        stratify=train['label'],

                                                                        random_state=42)
X_train = X_train.drop('label', axis = 1)

X_test = X_test.drop('label', axis=1)



X_train.head()
X_train = X_train.as_matrix()

X_test = X_test.as_matrix()

y_train = y_train.as_matrix()

y_test = y_test.as_matrix()


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
print(np.min(X_train))

print(np.max(X_train))

print(np.mean(X_train))

print(np.std(X_train))



print(np.min(X_test))

print(np.max(X_test))

print(np.mean(X_test))

print(np.std(X_test))
X_train = (X_train - 33.434) / 78.7047

X_test = (X_test - 33.434) / 78.7047
print(np.min(X_train))

print(np.max(X_train))

print(np.mean(X_train))

print(np.std(X_train))



print(np.min(X_test))

print(np.max(X_test))

print(np.mean(X_test))

print(np.std(X_test))
X_train[0]
from keras.layers import Conv2D, Input, Dense, SpatialDropout2D, Flatten

from keras.models import Model



input_layer = Input(shape=(28,28, 1))

conv_layer = Conv2D(filters = 64, kernel_size = (3,3), strides=(1, 1), padding='valid')(input_layer)

dropout = SpatialDropout2D(0.5)(conv_layer)

dense_layer = Dense(64, activation = 'relu')(dropout)

flatten = Flatten()(dense_layer)

output = Dense(10, activation = 'softmax')(flatten)

model = Model([input_layer], output)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.summary()
#didn't divde by 255

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data = (X_test, y_test))
X_sub = pd.read_csv('../input/test.csv')

X_sub.head()





X_sub = X_sub.as_matrix()

X_sub = X_sub.reshape(X_sub.shape[0], 28, 28, 1).astype('float32')

X_sub = (X_sub - 33.434) / 78.7047





print(np.min(X_sub))

print(np.max(X_sub))

print(np.mean(X_sub))

print(np.std(X_sub))
predictions = model.predict(X_sub, verbose=0)
# select the indix with the maximum probability

results = np.argmax(predictions,axis = 1)



results = pd.Series(results,name="Label")

results.head()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)