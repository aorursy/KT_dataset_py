import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import datasets
from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical
df = pd.read_csv('../input/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv')

n = len(df.index)

X = np.array(df.drop(['label'],axis = 1))

X = X.reshape(n,28,28,3)

y = np.array(df['label'])

labels = range(7)
plt.figure(figsize=(10,20))

for i in range(49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(X[i], cmap="gray_r")

    plt.title(labels[y[i]])
X = X/255

y_cat = to_categorical(y)

num_classes = y_cat.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=1)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(7,activation = 'softmax'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=20, batch_size = 200, verbose=1)