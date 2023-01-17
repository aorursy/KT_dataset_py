# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sbn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')



test_df = pd.read_csv('../input/test.csv')
# train_df.head()
# test_df.head()
# train_df.describe()
# train_df['label'].value_counts()
X_train = train_df.iloc[:, 1:].values

y_train = train_df['label'].values

# print(X_train.shape)

# print(y_train.shape)



X_test = test_df.values
# print(X_test.shape)
# from sklearn.model_selection import train_test_split

# X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
# X_val2.shape

# import random

# i = random.randint(1, 42000) # to select a random  row

# plt.imshow(X_train[i, :].reshape((28, 28))) # converting 784 pixels into 28 x 28 matrix and viewing it as an image

# print(train_df['label'][i])
X_train = X_train/255

X_test = X_test/255

# X_val2 = X_val2/255
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))



X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

# X_val2 = X_val2.reshape(X_val2.shape[0], *(28, 28, 1))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
cnn = Sequential()



cnn.add(Conv2D(64, (3,3), input_shape=(28, 28, 1), activation='relu'))



cnn.add(MaxPooling2D(pool_size=(2,2))) # pooling



cnn.add(Dropout(0.25)) # to prevent overfitting



cnn.add(Flatten())



# cnn.add(Dense(units=128, activation='relu'))

cnn.add(Dense(units=64, activation='relu'))



cnn.add(Dense(units=10, activation='softmax'))
# cnn.summary()
cnn.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, batch_size=512, epochs=150)
y_pred = cnn.predict(X_test)

# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# cnn.evaluate(X_test, y_val2)



# cm =confusion_matrix(y_val2, y_pred)
# y_pred
y_pred2=np.argmax(y_pred,axis=1)
# y_pred2.shape
# cm = confusion_matrix(y_val2, y_pred2)
# plt.figure(figsize=(14, 10))

# sbn.heatmap(cm, annot=True)

# print(classification_report(y_val2, y_pred2, target_names=["Class {}".format(i) for i in range(10)]))
index = np.array([i for i in range(1, y_pred2.shape[0]+1)])



# index.shape == y_pred2.shape
# index[-1]
sub = pd.DataFrame({'ImageId': index,'Label': y_pred2})
# sub.head()
# # import random

# i = random.randint(1, 28000) # to select a random  row

# plt.imshow(X_test[i, :].reshape((28, 28))) # converting 784 pixels into 28 x 28 matrix and viewing it as an image

# print(sub['Label'][i])
sub.to_csv('cnn1.csv', index=False)