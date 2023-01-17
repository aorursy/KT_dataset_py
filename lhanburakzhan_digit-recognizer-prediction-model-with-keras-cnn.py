# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
ss = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
train.head()
plt.figure(figsize=(20,5))
sns.countplot(train['label']);
train["label"].value_counts()
X = train.drop(columns=["label"]).copy()
y = train["label"]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.10, random_state=42)
plt.figure(figsize=(20,5))
for i in range(1,15):
    plt.subplot(2,7,i)
    img = X_train.sample(n=1).to_numpy().reshape(28,28)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    
plt.show()
X_train = X_train / 255
X_val = X_val / 255
test = test / 255
X_train  = X_train.to_numpy().reshape(X_train.shape[0], 28,28,1)
X_val = X_val.to_numpy().reshape(X_val.shape[0], 28,28,1)
test = test.to_numpy().reshape(test.shape[0], 28,28,1)
plt.figure(figsize=(20,20))
sns.heatmap(test[333].reshape(28,28), annot=True, cmap="gray")
plt.show()
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', input_shape = (28,28,1)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same'))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Dense(10, activation = "softmax"))

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 30  # for better result increase the epochs
batch_size = 512

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])

history.history.keys()
plt.plot(history.history["val_loss"], label="Test loss")
plt.plot(history.history["loss"], label="Training loss")
plt.legend()
plt.show()
plt.plot(history.history["val_accuracy"], label="Test Accuracy")
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.legend()
plt.show()
plt.plot(history.history["lr"], label="Learning Rate")
plt.legend()
plt.show()
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
y_val_labels = np.argmax(y_val, axis=1)
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(16,9))
sns.heatmap(confusion_matrix(y_val_labels, y_pred_labels), annot=True, fmt=".1f")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.title("Confusion matrix of the model")
plt.show()
from random import randint

plt.figure(figsize=[20,30])
for i in np.arange(1,20,2):
    random_number = randint(0,X_val.shape[0])
    plt.subplot(5,4,i)
    img = X_val[random_number].reshape(28,28)
    plt.imshow(img, cmap="gray")
    plt.title("True: " + str(y_val_labels[random_number]))
    plt.axis("off")
    
    #Arange plot color green if predicted correct or red if predicted wrong
    if y_val_labels[random_number] == y_pred_labels[random_number]:
        color="g"
    else:
        color="r"
    
    plt.subplot(5,4,i+1)
    plt.bar([i for i in range(0,10)], y_pred[random_number], color=color)
    plt.title("Predicted: " + str(y_pred_labels[random_number]))
    plt.xticks([i for i in range(0,10)])
    
plt.show()
wrong = np.where((y_pred_labels != y_val_labels) == True)[0]

from random import randint

plt.figure(figsize=[20,30])
for i in np.arange(1,20,2):
    random_number = int(np.random.choice(wrong, 1))
    plt.subplot(5,4,i)
    img = X_val[random_number].reshape(28,28)
    plt.imshow(img, cmap="gray")
    plt.title("True: " + str(y_val_labels[random_number]))
    plt.axis("off")
    
    #Arange plot color green if predicted correct or red if predicted wrong
    if y_val_labels[random_number] == y_pred_labels[random_number]:
        color="g"
    else:
        color="r"
    
    plt.subplot(5,4,i+1)
    plt.bar([i for i in range(0,10)], y_pred[random_number], color=color)
    plt.title("Predicted: " + str(y_pred_labels[random_number]))
    plt.xticks([i for i in range(0,10)])
    
plt.show()
y_test = y_pred = model.predict(test)
y_test_labels = np.argmax(y_test, axis=1)
submission = pd.DataFrame(y_test_labels)
submission.index += 1 #ImageId starting from 1
submission.index.name = "ImageId"
submission.columns = ["Label"] #name columns
submission.to_csv("submission.csv") # to csv