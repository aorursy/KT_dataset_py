%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_data = train_df.values
test_data = test_df.values
labels = train_data[:,0]
train = train_data[:,1:]/255
dummy_y = keras.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(train, dummy_y, test_size=0.1, random_state=166,stratify=labels)
x_train = x_train.reshape(x_train.shape[0],28,28, 1)
x_test = x_test.reshape(x_test.shape[0],28,28, 1)
model = Sequential()
callbacks = [keras.callbacks.ModelCheckpoint('minist.h5', monitor='val_acc', verbose=1, save_best_only=True,
                            mode='auto')]
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',padding='same',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
model.add(Conv2D(128, (28, 28),activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_data=(x_test, y_test),callbacks=callbacks)
model.load_weights('minist.h5')

test = test_data.reshape(test_data.shape[0],28,28, 1)/255
predict = model.predict(test)
results = np.argmax(predict,axis = 1)
submission = pd.DataFrame({"ImageId":range(1,28001),"Label":results})

submission.to_csv("cnn_mnist.csv",index=False)
submission