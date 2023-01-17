import pandas as pd

import os

path = '/kaggle/input/berkeley-multimodal-human-action-database/'



full_data = pd.DataFrame()



for entry in sorted(os.listdir(path)):

    if os.path.isfile(os.path.join(path, entry)):

        if entry.endswith('.txt'):

            data = pd.read_csv(path+entry,sep=' ',header=None)

            data.drop([129,130],inplace=True,axis=1)

            data['classs'] = entry[-10:-8]

            full_data = pd.concat([full_data,data],ignore_index=True)
full_data.shape
full_data.dtypes
full_data.head()
full_data.info()
x = full_data.drop(["classs"],axis=1)

y = full_data.classs.values

x.head()
y = pd.DataFrame(y)

y.iloc[:,0] = y.iloc[:,0].str.replace('t','1')

y.iloc[:,0] = y.iloc[:,0].str.replace('-','2')

y.astype('int32')
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)

print('Shape of train data is : ',x_train.shape)

print('Shape of label data is : ',y_train.shape)
from keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
from keras.callbacks import EarlyStopping

from tensorflow import keras



early_stop = EarlyStopping(monitor='loss', patience=2)

model = keras.Sequential()



model.add(keras.layers.Dense(128, activation='relu', input_shape=(129,)))



model.add(keras.layers.Dense(256, activation='relu'))



model.add(keras.layers.Dense(128, activation='relu'))



model.add(keras.layers.Dense(256, activation='relu'))



model.add(keras.layers.Dense(128, activation='relu'))



model.add(keras.layers.Dense(64, activation='relu'))



model.add(keras.layers.Dense(128, activation='relu'))



model.add(keras.layers.Dense(256, activation='relu'))



model.add(keras.layers.Dense(128, activation='relu'))



model.add(keras.layers.Dense(64, activation='relu'))



model.add(keras.layers.Dense(13, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
hist = model.fit(x_train , y_train , epochs=6, validation_split=0.20, batch_size= 128,callbacks=[early_stop])
from matplotlib import pyplot as plt

print(hist.history.keys())



plt.plot(hist.history['loss'],label = 'Train loss')

plt.plot(hist.history['val_loss'],label = 'Val loss')

plt.legend()

plt.show()
hist2 = model.evaluate(x_test,y_test)