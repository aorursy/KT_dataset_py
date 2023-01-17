!pip install keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.utils import np_utils
import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
labels = train.pop('label')
train_data = np.array((train / 255).values)
train_data.shape
train_data = train_data.reshape(42000, 28, 28, 1)
test_num = int(len(train_data)/7) # 6:1でデータを分ける
X_train, X_test = train_data[:-test_num], train_data[-test_num:]
X_train.shape, X_test.shape
labels.shape
y_train, y_test = labels[:-test_num], labels[-test_num:]
y_train.shape
CLASSES = 10
Y_train = np_utils.to_categorical(y_train, CLASSES)
Y_test = np_utils.to_categorical(y_test, CLASSES)
Y_train.shape, Y_test.shape
# モデルの定義
model = Sequential()

model.add(Conv2D(28,3,input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(28,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(56,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(1.0))

model.add(Dense(CLASSES, activation='softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])
BATCH_SIZE = 100
EPOCH = 5
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, validation_split=0.1)
test = pd.read_csv('../input/test.csv')
test.values.shape
val_data = test.values.reshape(28000, 28, 28, 1)
predicts = model.predict_classes(val_data)
output_data = pd.DataFrame({'ImageId': range(1, len(predicts)+1),'Label': predicts})
output_data.to_csv('submission.csv', index=False)
!zip submission.csv.zip submission.csv
ll 
!kaggle competitions submit -c digit-recognizer -f submission.csv.zip -m "CNN using Keras"
