import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout

# digits = pd.read_csv("../input/digit-recognizer/train.csv") # 1170th
digitsTest = pd.read_csv("../input/digit-recognizer/test.csv")
digits = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
digits_ext.head()
digits.head()
digits.shape, digitsTest.shape, digits_ext.shape
train = digits.values[:,1:]
trainLabels = digits.values[:,0]
test = digitsTest.values
train.shape, trainLabels.shape, test.shape
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(train[i].reshape(28,-1),cmap="gray")
    plt.title("Digit: {}".format(trainLabels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
plt.close()
train = train.astype("float32")
train/=255.0

trainLabels[:10]
num_categories = 10
trainLabels = keras.utils.to_categorical(trainLabels,num_categories)

trainLabels.shape
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense


train.shape
model = Sequential()
model.add(Conv2D(32, 
                 kernel_size=(3,3), 
                 activation = "relu", 
                 input_shape = (28,28,1)
))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation = "relu"))

model.add(Dense(128, activation = "relu"))

model.add(Dropout(0.5))
model.add(Dense(num_categories, activation = "softmax"))

#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10) 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


batch_size = 128
num_epoch = 15
#model training

model_log = model.fit(train.reshape(-1, 28,28,1), trainLabels,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          
          )
ypred = model.predict(test.reshape(-1,28,28,1))
import os
# plotting the metrics
fig = plt.figure()
# plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.tight_layout()
fig
ypred = np.argmax(ypred, axis = 1)
plt.imshow(test[0].reshape(-1,28), cmap="gray")
print(ypred[0])

import matplotlib.pyplot as plt
fig = plt.figure(1)
plt.figure(figsize=(12,12))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(test[i].reshape(28,-1),cmap="gray")
    plt.title("Digit: {}".format((ypred[i])))
    plt.xticks([])
    plt.yticks([])
plt.show()
plt.close()
ypreddf = pd.DataFrame(ypred)
ypreddf.columns = ["Label"]
ypreddf.head()

ypredDF = []
for i in range(len(ypred)):
    ypredDF.append([i+1, ypred[i]])
ypredDF = pd.DataFrame(ypredDF, columns=["ImageId","Label"])
ypredDF.to_csv("submissions.csv", index = False)
pd.read_csv("submissions.csv")