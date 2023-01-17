import os

os.chdir('../input/digit-recognizer')
!ls -al
import pandas
import keras

train_csv = pandas.read_csv('train.csv')

#X_train = train_csv.values.reshape(-1,28,28,1)
#X_final = train_csv.values.reshape(-1,28,28,1)
y_train = train_csv["label"]
x_train = train_csv.drop(labels = ["label"],axis = 1)

print(x_train.values.reshape(-1,28,28,1))
x_train = x_train.values.reshape(-1,28,28,1) / 225
y_train = keras.utils.to_categorical(y_train)
print(x_train.shape)
print(y_train.shape)
from keras.layers import Dense,Conv2D,Flatten,Dropout,BatchNormalization,Conv1D,InputLayer
from keras.models import Sequential

model = Sequential()
model.add(InputLayer(input_shape=(28,28,1)))
model.add(Conv2D(32,(4,4)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='softmax'))
model.add(Flatten())
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,batch_size=256,epochs=16)
model.save('/kaggle/working/digit_recognizer_cnn.h5')
import pandas

test_csv = pandas.read_csv('test.csv')

#print(test_csv.values.reshape(-1,28,28,1))
x_train = test_csv.values.reshape(-1,28,28,1) / 225


data = model.predict_classes(x_train)
#pandas.read_csv('sample_submission.csv')
import numpy as np
import time
predictiondf = pandas.DataFrame({'ImageId':np.arange(1,28001),'Label':data})
predictiondf.to_csv('/kaggle/working/prediction.csv',index=False)
from IPython.display import FileLink, FileLinks
os.chdir('/kaggle/working/')
FileLink('prediction.csv')
FileLink('digit_recognizer_cnn.h5')