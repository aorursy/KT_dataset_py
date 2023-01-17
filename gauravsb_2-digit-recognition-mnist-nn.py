import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# training data:

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train_data.head()

# creating training img and label

train_label = train_data["label"]

train_img = train_data.iloc[:30000,1:]

print(train_img.shape)

train_label.head()
# encoding training labels : one hot encoding

import keras

n_classes = 10

train_label_new = keras.utils.to_categorical(train_label, n_classes)

print(train_label_new.shape)

train_label_ = train_label_new[:30000,:]

train_label_.shape

# creating validation set

train_img_val = train_data.iloc[30000:,1:]

train_label_val = train_label_new[30000:,:]

print(train_img_val.shape)

print(train_label_val.shape)

# test data 

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

# print(test_data.head())

print(test_data.shape)
# importing necessary dependencies

from keras.models import Sequential

from keras.layers import Dense
# creating model with dense layer

model = Sequential()

model.add(Dense(512,input_shape=(784,), activation = "relu"))

model.add(Dense(512, activation = "relu"))

model.add(Dense(512, activation = "relu"))

model.add(Dense(10, activation = "softmax"))



# compiling model

model.compile(optimizer = "rmsprop",loss = "categorical_crossentropy", metrics = ['accuracy'])
model.fit(train_img,train_label_,epochs = 50, batch_size=128, validation_data = (train_img_val,train_label_val),shuffle=True)
predictions = model.predict(test_data)

result = [x.argmax() for x in predictions]

print(result)

output = pd.DataFrame({'ImageId':test_data.index +1,'Label': result})

print(output.head())

output.to_csv("/kaggle/working/submission6.csv",index = False)

print("Your submission was successfully saved!")
