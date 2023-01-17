# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras import Sequential

from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

data.head()

#Read the data, and format it better 

Y_data = np.array(data['label'])

Y_data = to_categorical(Y_data)

data = data.drop('label',axis=1)

X_data = data.as_matrix()

X_data = np.reshape(X_data,(len(X_data),28,28,1))

print(np.array(X_data).shape)
def build_keras_model():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model
train_X, test_X, train_Y, test_Y = train_test_split(X_data, Y_data, train_size=0.6)
model = build_keras_model()

print(model.input.shape)

print(train_X.shape)

model.summary()

a = model.fit(train_X, train_Y)

print("Test accuracy:{}".format(model.test_on_batch(test_X,test_Y)[1]))
data = pd.read_csv("../input/test.csv")

X_data = data.as_matrix()

X_data = np.reshape(X_data,(len(X_data),28,28,1))

output = pd.DataFrame(columns=["ImageId","Label"])

output['Label'] = np.argmax(model.predict(X_data),axis=1)

output['ImageId'] = range(1, 1 + len(X_data))

output.to_csv("output.csv",index=False)