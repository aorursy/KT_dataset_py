from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
num_classes =10
batch_size=128
epochs=30
train_data = pd.read_csv('../input/train.csv')
predict_data = pd.read_csv('../input/test.csv')
train_data.head()
img_rows,img_cols = 28,28
X,y = train_data.iloc[:,1:], train_data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
X_train = X_train.values.reshape(X_train.values.shape[0],img_rows,img_cols,1)
X_test = X_test.values.reshape(X_test.values.shape[0],img_rows,img_cols,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train,num_classes=num_classes)
y_test =  keras.utils.to_categorical(y_test,num_classes=num_classes)
input_shape = (img_rows,img_cols,1)
#Define model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(optimizer=keras.optimizers.Adadelta(),loss= keras.losses.categorical_crossentropy,metrics = ['accuracy'])
my_callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode=max)]
#hist = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,callbacks=my_callbacks,validation_data = (X_test,y_test))
hist = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data = (X_test,y_test))
score = model.evaluate(X_test,y_test,verbose=1)
print('Test Loss :' , score[0])
print('Test Accuracy :' , score[1])
epoch_list = list(range(1,len(hist.history['acc']) + 1))
plt.plot(epoch_list,hist.history['acc'],epoch_list,hist.history['val_acc'])
plt.legend(('Training Accuracy','Validaiton Accuracy'))
plt.show()
X_predict = predict_data.values.reshape(predict_data.values.shape[0],img_rows,img_cols,1)
result = model.predict_classes(X_predict,batch_size=batch_size,verbose=1)
df_result = pd.DataFrame()
df_result['ImageId'] = [i for i in range(1,28001)]
df_result['Label'] = list(result)
df_result.to_csv('results.csv',index=False)
