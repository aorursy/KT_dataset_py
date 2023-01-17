# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv(r"../input/digit-recognizer/train.csv")

test_df = pd.read_csv(r"../input/digit-recognizer/test.csv")

print("train_df shape:", train_df.shape)

print("test_df shape:", test_df.shape)
train_df.head()
test_df.head()
train_df.describe()
sns.countplot(train_df.iloc[:,0])
plt.figure(figsize=(16,6))

for i in range(40):

    plt.subplot(4,10,i+1)

    plt.axis("off")

    plt.title("label " + str(train_df.iloc[i,0]))

    plt.imshow(train_df.iloc[i,1:].values.reshape(28,28), cmap="gray")

plt.tight_layout()

X_train = train_df.iloc[:,1:]

y_train = train_df.iloc[:,0]

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:,1:], train_df.iloc[:,0], test_size=0.2)

X_train_org = X_train

X_test_org = X_test

standardScaler = StandardScaler()

standardScaler.fit(X_train)

X_train = standardScaler.transform(X_train)

X_test = standardScaler.transform(X_test)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

from sklearn.ensemble import RandomForestClassifier

randomForestClassifier = RandomForestClassifier()

randomForestClassifier.fit(X_train, y_train)

randomForestClassifier.score(X_test, y_test)
from sklearn.tree import DecisionTreeClassifier

decisionTreeClassifier = DecisionTreeClassifier()

decisionTreeClassifier.fit(X_train, y_train)

print(decisionTreeClassifier.score(X_test, y_test))
X_train_org = np.asarray(X_train_org).reshape(-1,28,28)

X_test_org = np.asarray(X_test_org).reshape(-1,28,28)

print(X_train_org.shape)

print(X_test_org.shape)
from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)
from keras.layers import Dense

from keras.layers import Dropout

from keras.models import Sequential

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy



model = Sequential()

model.add(Dense(250, activation="relu" ,input_shape=(784,)))

model.add(Dropout(0.25))

model.add(Dense(100, activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
history = model.fit(X_train, y_train, batch_size=50 ,epochs=4, validation_data=(X_test, y_test))
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)

print(X_train.shape)

print(X_test.shape)
from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten



modelCNN = Sequential()

modelCNN.add(Convolution2D(32, (3,3), strides=(1,1), padding="same", input_shape=(28, 28, 1), activation="relu" ))

modelCNN.add(Convolution2D(32, (3,3), strides=(1,1), padding="same", activation="relu" ))

modelCNN.add(MaxPooling2D(pool_size=(2,2), padding="same"))

modelCNN.add(Dropout(0.25))



modelCNN.add(Convolution2D(64, (3,3), strides=(1,1), padding="same", activation="relu" ))

modelCNN.add(Convolution2D(64, (3,3), strides=(1,1), padding="same", activation="relu" ))

modelCNN.add(MaxPooling2D(pool_size=(2,2), padding="same"))

modelCNN.add(Dropout(0.25))



modelCNN.add(Flatten())

modelCNN.add(Dense(128, activation="relu"))

modelCNN.add(Dropout(0.5))

modelCNN.add(Dense(10, activation="softmax"))

modelCNN.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"])

modelCNN.summary()
from keras.utils import plot_model

plot_model(modelCNN, show_shapes=True, show_layer_names=False, to_file='model.png')
from keras.models import load_model



historyCNN = modelCNN.fit(X_train, y_train, batch_size=30, epochs=5, validation_data=(X_test, y_test))

#modelCNN.save('modelCNN.h5')   # HDF5 file, you have to pip3 install h5py if don't have it

#modelCNN = load_model('modelCNN.h5')

#modelCNN.summary()
result = modelCNN.predict_classes(X_test, batch_size=30)

print("Key:", historyCNN.history.keys())

plt.plot(historyCNN.history["val_accuracy"], "ro-", label="val_accuracy")

plt.plot(historyCNN.history["accuracy"], "bo-", label="train_accuracy")

plt.legend()
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(np.argmax(y_test,axis=1), result)

plt.figure(figsize=(12,8))

sns.heatmap(confusion, annot=True, fmt="d")
pd.crosstab(np.argmax(y_test,axis=1), result, rownames=["label"], colnames=["predict"])
errorAddress = ~np.equal(np.argmax(y_test,axis=1), result)

print("Error number =", errorAddress.sum())

plt.figure(figsize=(12,4))

for i in range(20):

    plt.subplot(2, 10, i+1)

    plt.axis("off")

    plt.title("correct:" + str(np.argmax(y_test[errorAddress][i]))+ "\n predict:"+ str(result[errorAddress][i]))

    plt.imshow(X_test_org[errorAddress][i].reshape((28,28)), cmap="gray")

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

datagen.fit(X_train)
transformParameters = {"theta":30}

plt.subplot(2,2,1)

plt.imshow(datagen.apply_transform(X_train_org[1].reshape(28,28,1), {"theta":30}).reshape(28,28), cmap="gray")



plt.subplot(2,2,2)

plt.imshow(datagen.apply_transform(X_train_org[1].reshape(28,28,1), {"flip_horizontal":True}).reshape(28,28), cmap="gray")



plt.subplot(2,2,3)

plt.imshow(datagen.apply_transform(X_train_org[1].reshape(28,28,1), {"flip_vertical":True}).reshape(28,28), cmap="gray")



plt.subplot(2,2,4)

plt.imshow(datagen.apply_transform(X_train_org[1].reshape(28,28,1), {"shear":0.9}).reshape(28,28), cmap="gray")

modelCNN.fit_generator(datagen.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train)/32, 

                       validation_data=(X_test, y_test), epochs=30, callbacks=[learning_rate_reduction])
result = modelCNN.predict_classes(X_test, batch_size=30)

errorAddress = ~np.equal(np.argmax(y_test,axis=1), result)

print("Error number =", errorAddress.sum())

plt.figure(figsize=(16,8))

for i in range(40):

    plt.subplot(4, 10, i+1)

    plt.axis("off")

    plt.title("correct:" + str(np.argmax(y_test[errorAddress][i]))+ "\n predict:"+ str(result[errorAddress][i]))

    plt.imshow(X_test_org[errorAddress][i].reshape((28,28)), cmap="gray")
#modelCNN.fit_generator(datagen.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train)/32, validation_data=(X_test, y_test), epochs=10)
submit_test_df = standardScaler.transform(test_df)

submit_test_df = submit_test_df.reshape(-1,28,28,1)

result = modelCNN.predict_classes(submit_test_df)
answer = pd.DataFrame()

answer["ImageId"] = np.arange(1,len(result)+1,1)

answer["Label"] = result

answer.to_csv("sumbit.csv")

answer