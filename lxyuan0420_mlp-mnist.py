import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.utils import to_categorical





import os

print(os.listdir("../input"))
# Data reading

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
trainLabel = to_categorical(train['label'], 10)

trainFeature = train.drop(columns = 'label')
# Normalising data

trainFeature /= 255.0

test /= 255.0
# Building Model

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(784,)))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.summary()

model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



hisotry = model.fit(trainFeature, trainLabel, epochs=30, batch_size=32)



pred_testLabel = model.predict(test)

#return index with the max prob.

testLabel = np.argmax(pred_testLabel, axis=1)

submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label': testLabel })

submission.to_csv("simpleMLP_mnist.csv",index=False)