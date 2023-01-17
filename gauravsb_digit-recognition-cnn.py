import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Loading train data

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train_data.head()

train_label = train_data["label"]

train_img = train_data.iloc[:,1:]

print(train_label)

print(train_img.head())
train_img.describe()
# Loading test data

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test_data.head()

from keras import layers

from keras.models import Sequential
model  = Sequential()

model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (28,28,1)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation = 'relu'))

model.add(layers.Dense(10,activation = 'softmax'))

model.summary()
# making lables categorical 

from keras.utils import to_categorical

train_label = to_categorical(train_label)

train_label.shape
# reshaping img dataframe to particular network input shape

train_img = train_img.values.reshape(-1,28,28,1)

train_img = train_img.astype('float32')/255

print(train_img.shape)

test_img = test_data.values.reshape(-1,28,28,1)

test_img = test_img.astype('float32')/255

print(test_img.shape)
# compiling model

model.compile(optimizer = 'rmsprop',

             loss = 'categorical_crossentropy',

             metrics = ['accuracy'])
# training Model

model.fit(train_img,train_label,epochs = 5, batch_size = 64)
predictions = model.predict(test_img)

result = [x.argmax() for x in predictions]

result
output = pd.DataFrame({'ImageId':test_data.index +1,'Label': result})

print(output.head())

output.to_csv("/kaggle/working/CNN_1.csv",index = False)

print("Your submission was successfully saved!")