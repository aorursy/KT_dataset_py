from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from keras import metrics

from keras.losses import categorical_crossentropy

from keras.utils.np_utils import to_categorical 



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import pandas as pd

import numpy as np
!ls ../input/digit-recognizer
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
test.head()
train.isnull().sum().describe()
test.isnull().sum().describe()
y_train = train['label']

x_train = train.drop(['label'], axis=1)
plt.hist(y_train, bins=10)
x_train = x_train / 255.0

test = test / 255.0
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)
xx_train, x_test, yy_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state=1)
model = Sequential()

model.add(Conv2D(input_shape=(28,28,1),filters=32,kernel_size=(3,3),padding="same", activation="relu"))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=10, activation="softmax"))
opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])
history = model.fit(xx_train, yy_train, epochs = 5, 

                    validation_data = (x_test, y_test), verbose = 1)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend()

plt.title('Loss Trend for training and validation data')

plt.ylabel('Epochs')

plt.xlabel('Loss')

plt.show()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend()

plt.title('Accuracy Trend for training and validation data')

plt.ylabel('Epochs')

plt.xlabel('Accuracy')

plt.show()
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="label")

results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission
submission.to_csv("cnn_mnist_submission.csv",index=False)