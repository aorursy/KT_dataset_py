import keras
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import RMSprop

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

np.random.seed(5)
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train[:5000]
y_train=y_train[:5000]
X_test=X_test[:5000]
y_test=y_test[:5000]
check_test=y_test
plt.imshow(X_train[1],cmap=plt.cm.binary)
conv_train=X_train
conv_test=X_test
X_train,X_test=X_train/255,X_test/255
X_train,X_test=X_train.reshape(5000,784),X_test.reshape(5000,784)
num_classes=10
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
epochs=60
learning_rate=0.1
decay_rate=learning_rate/epochs
momentum=0.8
sgd=SGD(learning_rate=learning_rate,momentum=momentum,decay=decay_rate,nesterov=False)
input_dim=X_train.shape[1]
lr_model=Sequential()
lr_model.add(Dense(64,activation='relu',kernel_initializer='uniform',input_dim=input_dim))
lr_model.add(Dropout(0.1))
lr_model.add(Dense(64,kernel_initializer='uniform',activation='relu'))
lr_model.add(Dense(num_classes,kernel_initializer='uniform',activation='softmax'))

lr_model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
batch_size=int(input_dim/100)
lr_model_history=lr_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test))
pred=lr_model.predict_classes(X_test)
test_main=pd.read_csv('../input/digit-recognizer/test.csv')
sample=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
prediction=lr_model.predict_classes(test_main)
prediction=pd.DataFrame(prediction)
prediction['ImageId']=sample['ImageId']
prediction.columns=['Label','ImageId']
prediction=prediction[['ImageId','Label']]
prediction.to_csv('submission',index=False)
def acc(y_test,pred):
    score=[]
    for i in range(len(pred)):
        if check_test[i]==pred[i]:
            score.append(1)
        else:
            score.append(0)
    return sum(score)/len(pred)
acc(check_test,pred)
epochs=60
learning_rate=0.1
decay_rate=0.1
momentum=0.8

sgd=SGD(lr=learning_rate,momentum=momentum,decay=decay_rate,nesterov=False)
input_dim=X_train.shape[1]
num_classes=10
batch_size=196


exponential_decay_model=Sequential()
exponential_decay_model.add(Dense(64,activation='relu',kernel_initializer='uniform',
                                 input_dim=input_dim))
exponential_decay_model.add(Dropout(0.1))
exponential_decay_model.add(Dense(64,kernel_initializer='uniform',activation='relu'))
exponential_decay_model.add(Dense(num_classes,kernel_initializer='uniform',activation='softmax'))
exponential_decay_model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['acc'])

def exp_decay(epoch):
    lrate=learning_rate * np.exp(-decay_rate*epoch)
    return lrate

loss_history=History()
lr_rate=LearningRateScheduler(exp_decay)
callbacks_list=[loss_history,lr_rate]
exponential_decay_history=exponential_decay_model.fit(X_train,y_train,
                                                      batch_size=batch_size,epochs=300,
                                                      callbacks=callbacks_list,verbose=1,
                                                      validation_data=(X_test,y_test))
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPool2D
conv_train=conv_train.reshape(conv_train.shape[0],28,28,1)
conv_test=conv_test.reshape(conv_test.shape[0],28,28,1)
test_2=test_main.values.reshape(test_main.shape[0],28,28,1)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.fit(conv_train,y_train,epochs=20,validation_data=(conv_test,y_test),batch_size=100)
submission2=exponential_decay_model.predict_classes(test_main)
submission2=pd.DataFrame(submission2)
submission2['ImageId']=sample['ImageId']
submission2.columns=['Label','ImageId']
submission2=submission2[['ImageId','Label']]
submission2.to_csv('exponential_model_submission.csv',index=False)
submission3=model.predict_classes(test_2)
submission3=pd.DataFrame(submission3)
submission3['ImageId']=sample['ImageId']
submission3.columns=['Label','ImageId']
submission3=submission3[['ImageId','Label']]
submission3.to_csv('conv_model_submission.csv',index=False)
submission2
test_main.values.shape
