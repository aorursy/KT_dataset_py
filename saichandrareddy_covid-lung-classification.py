# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/corona-lungs-dataset/COVID.csv")
data.info()
data.shape
data.head()
data.columns
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
data = data.set_index(['Unnamed: 0.1.1'])
X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X  = sc.fit_transform(X)
import matplotlib.pyplot as plt



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))

ax1.imshow(X[1].reshape(224, 224, 3).astype(float))

ax2.imshow(X[-1].reshape(224, 224, 3).astype(float))

ax1.title.set_text("label :-"+y[1])

ax2.title.set_text("Label :-"+y[-1])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=54)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=54)
X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape
X_train = X_train.reshape(len(X_train), 224, 224, 3).astype(float)

X_test = X_test.reshape(len(X_test), 224, 224, 3).astype(float)

X_val = X_val.reshape(len(X_val), 224, 224, 3).astype(float)
from keras.utils import to_categorical
lb = LabelEncoder()



y_train = lb.fit_transform(y_train)

y_test = lb.fit_transform(y_test)

y_val = lb.fit_transform(y_val)
y_train = to_categorical(y_train).astype(float)

y_test = to_categorical(y_test).astype(float)

y_val = to_categorical(y_val).astype(float)
from keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPool2D

from keras.models import Sequential

from keras.optimizers import Adam
model = Sequential()



model.add(Conv2D(8, kernel_size=(5, 5), strides=(1,1), padding="valid", activation="relu", input_shape=(224, 224, 3)))

model.add(MaxPool2D((4, 4)))

model.add(Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding="valid", activation="relu"))

model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(2, 2), strides=(1,1), padding="valid", activation="relu"))

model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, kernel_size=(2, 2), strides=(1,1), padding="valid",activation="relu"))

model.add(Dropout(0.5))

model.add(Flatten())



model.add(Dense(2, activation="softmax"))
model.summary()
from keras.preprocessing import image



train = image.ImageDataGenerator(rescale=1/255,

                                fill_mode="nearest")
batch_size = 8

epochs = 30
optimizer = Adam(lr = 0.001, decay=0.001/epochs)
model.compile(loss="binary_crossentropy", 

             optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train.flow(X_train, y_train, batch_size=batch_size), epochs =epochs,steps_per_epoch=len(X_train)//batch_size,

                   validation_data=train.flow(X_val, y_val), validation_steps=len(X_val)//batch_size)
model.evaluate(x=X_test, y=y_test)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
total = sum(sum(cm))

acc = (cm[0, 0] + cm[1, 1]) / total

sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity

print(cm)

print("acc: {:.4f}".format(acc))

print("sensitivity: {:.4f}".format(sensitivity))

print("specificity: {:.4f}".format(specificity))
y_pred_cat = []

y_test_cat = []



def cat(array, predit):

    for i in range(len(array)):

        if array[i] == 0:

            predit.append("covid")

        else:

            predit.append("normal")

cat(y_pred, y_pred_cat)

cat(y_test, y_test_cat)



print(y_pred_cat)
fig, ((ax1, ax2, ax3), (ax4,ax5, ax6)) = plt.subplots(2,3, figsize=(16, 16))



ax1.imshow(X_test[0])

ax1.title.set_text("predicted- "+y_pred_cat[0]+" Original:-"+y_test_cat[0])

ax2.imshow(X_test[1])

ax2.title.set_text("predicted- "+y_pred_cat[1]+" Original:-"+y_test_cat[1])

ax3.imshow(X_test[2])

ax3.title.set_text("predicted- "+y_pred_cat[2]+" Original:-"+y_test_cat[2])

ax4.imshow(X_test[3])

ax4.title.set_text("predicted- "+y_pred_cat[3]+" Original:-"+y_test_cat[3])

ax5.imshow(X_test[4])

ax5.title.set_text("predicted- "+y_pred_cat[4]+" Original:-"+y_test_cat[4])

ax6.imshow(X_test[5])

ax6.title.set_text("predicted- "+y_pred_cat[5]+" Original:-"+y_test_cat[5])