import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage import data,io,filters
from sklearn import preprocessing

from keras import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import cv2
df = pd.read_csv("../input/train.csv")
X = (df.iloc[:,1:].values/255).reshape(-1,28,28)
X = np.expand_dims(X, axis=3)
y = df['label'].values
enc = preprocessing.OneHotEncoder()
enc.fit(y.reshape(-1,1))
y = enc.transform(y.reshape(-1,1)).toarray()
def dieCNN():
    model = Sequential()
    model.add(Conv2D(32,(5,5), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32,(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,(3,3), strides=(3,3), activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    
    model.add(Dense(10, activation='softmax'))
    
    return model
model = dieCNN()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X,y, epochs=150, verbose=1, batch_size=32)
model.save_weights("model.h5")
X_test = pd.read_csv("../input/test.csv").values
pred = pd.read_csv("../input/sample_submission.csv").values
for i,x in enumerate(X_test):
    tmp = (x/255).reshape(-1,28,28)
    tmp = np.expand_dims(tmp, axis=3)
    pred[i,1] = np.argmax(model.predict(tmp))

output = pd.DataFrame(pred)
output.to_csv("output.csv",header=['ImageId','Label'],index=False)
