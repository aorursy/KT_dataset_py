import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

sns.set()

ds = pd.read_csv("../input/A_Z Handwritten Data/A_Z Handwritten Data.csv")
ds.rename(columns={'0':'alph'}, inplace=True)


X = ds.drop('alph',axis = 1)
y = ds['alph']
print(y)
print("shape:",X.shape)
plt.figure(figsize = (12,10))
plt.imshow(X.iloc[30000].values.reshape(28,28))
plt.show()
print("No. of each letters")

letter_map = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
#print(dataset['label'])
ds['alph'] = ds['alph'].map(letter_map)
temp=ds['alph'].value_counts().reset_index()
plt.bar(x=temp['index'],height=temp['alph'])
plt.show()

xtr, xts, ytr, yts = train_test_split(X,y)
standard_scaler = StandardScaler()
standard_scaler.fit(xtr)

xtr = standard_scaler.transform(xtr)
xts = standard_scaler.transform(xts)

import matplotlib.pyplot as plt
print('After standardizing')
plt.figure(figsize = (12,10))
plt.imshow(xtr[0].reshape(28,28))
plt.show()
xtr = xtr.reshape(xtr.shape[0], 28, 28, 1).astype('float32')
xts = xts.reshape(xts.shape[0], 28, 28, 1).astype('float32')

ytr = np_utils.to_categorical(ytr)
yts = np_utils.to_categorical(yts)
xtr.shape
ytr.shape
cls = Sequential()
cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
cls.add(MaxPooling2D(pool_size=(2, 2)))
cls.add(Dropout(0.3))
cls.add(Flatten())
cls.add(Dense(128, activation='relu'))
cls.add(Dense(len(y.unique()), activation='softmax'))

cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = cls.fit(xtr, ytr, validation_data=(xts, yts), epochs=18, batch_size=200, verbose=2)

scores = cls.evaluate(xts,yts, verbose=0)
print("CNN Score:",scores[1])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss during training')
plt.xlabel('Epoch Number')
plt.legend(['Train', 'Test'], loc='upper middle')
plt.show()
cls.evaluate(X_train,y_train)
cls.evaluate(X_test,y_test)
cls.summary()


