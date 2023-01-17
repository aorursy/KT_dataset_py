import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
batch_size = 128
num_classes = 10 
epochs = 2 
train = pd.read_csv('../input/digit-recognizer/train.csv').values
trainY = np_utils.to_categorical(train[:,0].astype('int32'), num_classes) # labels
trainX = train[:, 1:].astype('float32') 
trainX /= 255 

rows, cols = 28, 28
trainX = trainX.reshape(trainX.shape[0], rows,cols, 1)
input_shape = (rows, cols, 1)
model = Sequential()
model.add(Conv2D(32,data_format='channels_last',kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

model.fit(trainX, trainY,batch_size=batch_size,epochs=epochs,verbose=2)
score = model.evaluate(trainX, trainY, verbose=2)
print('Train accuracy:', score[1])
testX = pd.read_csv('../input/digit-recognizer/test.csv').values.astype('float32')
testX /= 255
testX = testX.reshape(testX.shape[0],rows,cols, 1)
testY = model.predict_classes(testX, verbose=2)
pd.DataFrame({"ImageId": list(range(1,len(testY)+1)),
              "Label": testY}
            ).to_csv('submission.csv', index=False, header=True)


