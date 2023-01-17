import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

epochs = 15
train_data.shape
test_data.shape
train_data.head()
test_data.head()
target_data = train_data.label.values
train_features = train_data.loc[:,train_data.columns !='label'].values
train_features
target_data

#normalizing the values

train_features = train_features.astype('float32')/255

test_data = test_data.astype('float32')/255
train_features = train_features.reshape(-1,28,28,1)

test = test_data.values.reshape(-1,28,28,1)
train_features.shape
test.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_features,target_data,test_size=0.1,random_state = 42)
y_train = keras.utils.to_categorical(y_train,10)

y_test = keras.utils.to_categorical(y_test,10)
print(y_train.shape,y_test.shape)
model = Sequential()

model.add(Conv2D(128,kernel_size=(5,5),activation='relu',input_shape= (28,28,1))) # 128 shows the numb of filters kernelsize gives the size of each filter

model.add(Conv2D(64,(5,5),activation='relu'))

model.add(Conv2D(32,(5,5),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #Pooling layer

model.add(Dropout(0.25))                  #Dropout layer

model.add(Flatten())                      #Flatten Layer

model.add(Dense(128, activation='relu'))  #Fully connected layer

model.add(Dropout(0.5))                   #Dropout layer

model.add(Dense(10, activation='softmax')) #Output Layer

#Model compilation

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
#Training the model

model.fit(x_train, y_train,

          batch_size=128,

          epochs=epochs,

          verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
y_predicted = model.predict(x_test)

y_pred_classes = np.argmax(y_predicted,axis=1)

y_act = np.argmax(y_test,axis=1)
y_pred_classes 
y_act
import sklearn.metrics as metrics
metrics.confusion_matrix(y_act,y_pred_classes)
predicted_test = model.predict(test)# predicting on test data

predicted_test_class = np.argmax(predicted_test,axis=1)
predicted_test_class
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
#submission_dataframe = pd.DataFrame({'ImageId': list(range(1,len(predicted_test_class)+1)),'Label':predicted_test_class})

#submission_dataframe.to_csv('../input/predict_submission.csv',index=False,header=True)
 