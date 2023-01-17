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
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping
BATCH_SIZE = 32

EPOCHS = 100

IMG_W = 28

IMG_H = 28
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input//Kannada-MNIST/test.csv')

digi = pd.read_csv('/kaggle/input//Kannada-MNIST/Dig-MNIST.csv')
def process_data(data, dims):

    X = data.iloc[:,1:].to_numpy(dtype="float32")

    y = data['label'].to_numpy()

    

    X = X/255

    X = X.reshape(-1, dims[0], dims[1], 1)

    y = to_categorical(y)

    

    print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

    

    return X,y
#train.head()

#test.head()

digi.head(3)
X_val, y_val = process_data(digi, dims=(IMG_W, IMG_H))
X_train, y_train = process_data(train, dims=(IMG_W, IMG_H))
model = Sequential()



model.add(Conv2D(128,(3,3),activation='relu', input_shape=(28,28,1)))

# model.add(BatchNormalization(momentum=0.2, gamma_initializer='uniform'))

# model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3),activation='relu'))

# model.add(BatchNormalization(momentum=0.1, gamma_initializer='uniform'))

model.add(MaxPool2D((2,2)))

# model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D((2,2)))

# model.add(BatchNormalization(momentum=0.1, gamma_initializer='uniform'))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(BatchNormalization(momentum=0.1, gamma_initializer='uniform'))

model.add(MaxPool2D((2,2)))

# model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=2, callbacks=[EarlyStopping(monitor='val_acc', patience=5)])

# model.fit(X_train, y_train, epochs=5, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=2)
X_test = test.iloc[:,1:].to_numpy(dtype="float32")



X_test = X_test.reshape(-1,IMG_W, IMG_W, 1)

X_test.shape
predictions_new = model.predict(X_test)

result = [np.argmax(pred) for pred in predictions_new]



# Save test predictions to file

output = pd.DataFrame({"id": test.id, "label": result})

output.to_csv("submission.csv", index=False)