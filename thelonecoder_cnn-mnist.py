import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
train_df = pd.read_csv('../input/digit-recognizer/train.csv')
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
X_train = (train_df.drop('label',axis=1)).values
y_train = train_df['label'].values
X_train = X_train.reshape((42000,28,28))
plt.imshow(X_train[0])
y_train[0]
X_test = test_df.values
X_test = X_test.reshape(((28000, 28,28)))
plt.imshow(X_test[0])
X_train = X_train /  255
X_test =  X_test /  255
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
from tensorflow.keras.utils import to_categorical
y_train_cat =to_categorical(y_train,10)
y_val_cat =to_categorical(y_val,10)
X_train = X_train.reshape((33600, 28, 28,1))
X_val = X_val.reshape((8400, 28, 28,1))
X_test = X_test.reshape((28000,28,28,1))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
early_stop = EarlyStopping(monitor='val_loss',patience=1)
with tf.device('/gpu:0'):
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])
    model.fit(x=X_train,y=y_train_cat,epochs=10,validation_data = (X_val,y_val_cat),
             callbacks=[early_stop])
   
losses_acc = pd.DataFrame(model.history.history)
losses_acc
losses_acc[['loss','val_loss']].plot()
losses_acc[['accuracy','val_accuracy']].plot()
model.metrics_names
model.evaluate(X_val,y_val_cat)
predictions = model.predict_classes(X_test)
predictions
predictions.shape
X_test.shape
plt.imshow(X_test[0].reshape((28,28)))
result = pd.DataFrame({'ImageId': np.arange(1,len(predictions)+1),'Label' : predictions})
result.to_csv(os.getcwd() + '/out.csv',index=False)

