import pandas as pd

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
y_train = train['label'].values

X_train = train.drop('label',1).values

X_test = test.values
print('y_train.shape:{}'.format(y_train.shape))

print('X_train.shape:{}'.format(X_train.shape))

print('X_test.shape:{}'.format(X_test.shape))
from keras import models,layers

network = models.Sequential()

network.add(layers.Dense(512,activation='relu',input_shape=(784,)))

network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='adam',

               loss='categorical_crossentropy',

               metrics=['accuracy'])
X_train = X_train.astype('float32') / 255

X_test = X_test.astype('float32') / 255
from keras.utils import to_categorical

y_train = to_categorical(y_train)
network.fit(X_train,y_train,epochs=20,batch_size=512)
import numpy as np

results = network.predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("begginer.csv",index=False)