import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data=pd.read_csv("../input/train.csv")

train_data.head(n=10)

from keras.utils import to_categorical

train_data_x= train_data.values[:,1:]
train_data_y= train_data.values[:,0]

train_data_x= train_data_x.reshape((-1,28,28,1)) 
train_data_y= to_categorical(train_data_y) 

print("Training Images : ",train_data_x.shape)
print("Training Labels : ",train_data_y.shape)

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import Adam

model=Sequential()

model.add(BatchNormalization(axis=-1,input_shape=(28,28,1)))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(filters=40, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))

model.add(Dropout(0.275))

model.add(Flatten())

model.add(BatchNormalization(axis=-1))

model.add(Dense(units=96,activation='relu'))

model.add(Dense(units=67,activation='relu'))

model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()
model.fit(train_data_x,train_data_y,epochs=30,batch_size=256, verbose=2)
test_data=pd.read_csv("../input/test.csv")

test_data.head(n=10)
train_data_x= test_data.values.reshape((-1,28,28,1)) 

predictions_proba=model.predict(train_data_x)

predictions_class=np.argmax(predictions_proba,axis=1).reshape((-1,1))
img_ids=np.array(range(1,predictions_class.shape[0]+1)).reshape((-1,1))
print(img_ids.shape)
output_frame=pd.DataFrame(data=np.hstack((img_ids,predictions_class)),columns=['ImageId','Label'])
output_frame.head(n=10)
output_frame.to_csv('predictions.csv')
print("Output file saved as predictions.csv")
