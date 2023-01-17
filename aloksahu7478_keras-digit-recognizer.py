import pandas as pd

from keras.preprocessing.image import load_img, array_to_img

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train_lbl = train['label']

train_data = train.drop(['label'],axis=1)
train_data = train_data.astype('float32')

test = test.astype('float32')
train_data /= 255

test /= 255
train_data = train_data.to_numpy()

test = test.to_numpy()
train_lbl = to_categorical(train_lbl,10)
model = Sequential()

model.add(Dense (900,activation='relu',input_shape = (784,)))

model.add(Dense(900,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam', loss ='categorical_crossentropy', metrics = ['accuracy'])

model.summary()
hist = model.fit(train_data,train_lbl,epochs=20)
prediction = model.predict(test)
output = prediction.astype('int32')
output2 = pd.DataFrame(output)
output2
for col in output2:

    output2[col] = output2[col].apply(lambda x : x * col)
new_output = output2.sum(axis=1)
new_output = pd.DataFrame(new_output)
new_output = new_output[0].rename('Label')

new_output.index.name = 'ImageId'
new_output.index += 1
new_output


new_output.to_csv('./Submission.csv')