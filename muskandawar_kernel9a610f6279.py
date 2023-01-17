# Run this cell to mount your Google Drive.

from google.colab import drive

drive.mount('/content/drive')



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

import matplotlib.pyplot as plt

df_train=pd.read_csv('drive/My Drive/train.csv')

df_test=pd.read_csv('drive/My Drive/test.csv')



dataset_train=np.array(df_train)

dataset_test=np.array(df_test)

y = df_train['label'].values

X=df_train.drop(['label'],axis=1).values

test=df_test.drop(['id'],axis=1).values



X_train = X / 255.0

X_test = test / 255.0

X_train = X.reshape(-1,28,28,1)

X_test = test.reshape(-1,28,28,1)

Y_train = to_categorical(y, num_classes = 10)
model = Sequential()





model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu',))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))



model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train,Y_train,batch_size=128,epochs=25)
predicted_classes = model.predict_classes(X_test)

print(len(predicted_classes))
writefp = open('cnn22.csv','w')

writefp.write('id,'+'pred'+'\n')

for i,a in enumerate(predicted_classes):

  writefp.write(str(i+1)+','+str(a)+'\n')

writefp.close()