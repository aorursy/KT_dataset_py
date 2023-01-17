import pandas as pd

import numpy as np

from tensorflow import keras

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras import Sequential

import os

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import BatchNormalization

from keras.layers import Conv2D

import seaborn as sns

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
df = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')
X = df.drop(['label'],axis=1)

y = df['label'].copy()
sns.countplot(y)
labels = to_categorical(y)

labels
X = X/255.0

df_test = df_test/255.0
X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.3,stratify=labels)
X = X.values.reshape((-1,28,28,1))
df_test = df_test.values.reshape((-1,28,28,1))
X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.3,stratify=labels)
datagen = ImageDataGenerator(rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1,)
datagen.fit(X)
model3 = Sequential()



model3.add(Conv2D(32,(5,5),strides = 1 , padding='same',activation='relu',input_shape=(28,28,1)),)

model3.add(BatchNormalization())

model3.add(Conv2D(32,(4,4),strides = 1 , padding='same',activation='relu'))

model3.add(BatchNormalization())

model3.add(Dropout(0.4))

model3.add(Conv2D(64,(4,4),strides = 1 , padding='same',activation='relu'))

model3.add(BatchNormalization())

model3.add(Conv2D(64,(4,4),strides = 1 , padding='same',activation='relu'))

model3.add(BatchNormalization())

model3.add(Dropout(0.4))

model3.add(Conv2D(128,(4,4),strides = 1 , padding='same',activation='relu'))

model3.add(Flatten())

model3.add(Dense(256,activation='relu'))

model3.add(Dense(10,activation='softmax'))



model3.compile(keras.optimizers.Adam(),keras.losses.categorical_crossentropy,['accuracy'])
print(model3.summary())
X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.3)
model3.fit_generator(datagen.flow(X_train,y_train,batch_size=100),epochs=10,validation_data=(X_test,y_test))
model3.fit_generator(datagen.flow(X,labels,batch_size=100),epochs=40)
y_hat = np.argmax( model3.predict(df_test) , axis = 1 )
res = pd.DataFrame()



res['Label'] = y_hat

res['ImageId'] = res.index.values + 1
res = res[['ImageId','Label']]

res.head()
res.to_csv('submission.csv',index=False)