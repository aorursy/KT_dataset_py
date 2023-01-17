import pandas as pd

import numpy as np



# 讀取csv

df_train = pd.read_csv('../input/train.csv', encoding = 'big5')



df_train[:5]
from sklearn.model_selection import train_test_split

from keras.utils import np_utils



X_train = df_train[df_train.columns[1:]].values

y_train = df_train['label'].values

#X_train_raw = df_train[df_train.columns[1:]].values

#y_train_raw = df_train['label'].values

#X_train, X_test, y_train, y_test = train_test_split(X_train_raw, y_train_raw, test_size=0.1)



# Input shape format: (28, 28, 1)

# If 128x128 RGB, (128,128,3)

X_train = X_train.reshape(X_train.shape[0],28,28,1) / 255

#X_test = X_test.reshape(X_test.shape[0],28,28,1) / 255



# one-hot encoding

y_train_oneHot = np_utils.to_categorical(y_train, num_classes=10)

#y_test_oneHot = np_utils.to_categorical(y_test, num_classes=10)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(12,3))

plt.plot(X_train[:1].reshape(-1))

plt.figure(figsize=(6,6))

plt.matshow(X_train[:1].reshape(28,28), cmap = plt.get_cmap('binary'))



y_train[:1]
from keras.models import Sequential

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization



# Conv + MaxPooling 1

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Conv + MaxPooling 2

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Flatten層: 壓成一維

# Dense 接在內層不用input_dim，其他參數先用預設值

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='normal'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu', kernel_initializer='normal'))

model.add(Dropout(0.5))



model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 

history=model.fit(X_train, y_train_oneHot, validation_split=0.2, epochs=20, batch_size=250, verbose=1)
import matplotlib.pyplot as plt

%matplotlib inline

def plot_train_history(history, train_metrics, val_metrics):

    plt.plot(history.history.get(train_metrics),'-o')

    plt.plot(history.history.get(val_metrics),'-o')

    plt.ylabel(train_metrics)

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'])
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

plot_train_history(history, 'loss','val_loss')

plt.subplot(1,2,2)

plot_train_history(history, 'acc','val_acc')
model.evaluate(X_train, y_train_oneHot)
# 讀取csv

df_test = pd.read_csv('../input/test.csv', encoding = 'big5')



X_test = df_test.values



# Input shape format: (28, 28, 1)

# If 128x128 RGB, (128,128,3)

X_test = X_test.reshape(X_test.shape[0],28,28,1) / 255



test_img = np.reshape(X_test[:1, :], (28, 28))

plt.matshow(test_img, cmap = plt.get_cmap('binary'))

plt.show()
prediction = model.predict_classes(X_test)



prediction[:1]
# 匯出預測

df = pd.DataFrame(prediction)

df.columns = ['Label']

df.index = np.arange(1, len(df)+1)



#df.to_csv("../input/submission.csv")