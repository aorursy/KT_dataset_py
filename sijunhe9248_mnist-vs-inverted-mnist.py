# LOAD LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

# LOAD THE DATA
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_test = test

X_train_inverted = 255 - X_train
X_test_inverted = 255 - X_test

Y_train = to_categorical(Y_train, num_classes = 10)

X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0 
X_test = X_test.values.reshape(-1, 28, 28, 1) / 255.0
X_train_inverted = X_train_inverted.values.reshape(-1, 28, 28, 1) / 255.0
X_test_inverted = X_test_inverted.values.reshape(-1, 28, 28, 1) / 255.0
plt.figure()
plt.subplot(121)
plt.imshow(X_train[0][:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(X_train_inverted[0][:,:,0], cmap='gray')
def create_model():
    model = Sequential()
    model.add(Conv2D(24,kernel_size=5,padding='same',activation='relu',
            input_shape=(28,28,1)))
    model.add(MaxPool2D())
    model.add(Conv2D(48,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
val_acc_normal = []
for run in range(10):
    model = create_model()
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.333)
    # TRAIN NETWORKS
    epochs = 20
    history = model.fit(X_train2,Y_train2, batch_size=128, epochs = epochs, 
            validation_data = (X_val2,Y_val2), verbose=0)
    val_acc_normal.append(max(history.history['val_acc']))
    print("Run {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(run + 1, epochs, max(history.history['acc']),max(history.history['val_acc']) ))
print("Average validation accuracy: {:.5f}".format(sum(val_acc_normal) / len(val_acc_normal)))
val_acc_inverted = []
for run in range(10):
    model = create_model()
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train_inverted, Y_train, test_size = 0.333)
    # TRAIN NETWORKS
    epochs = 20
    history = model.fit(X_train2,Y_train2, batch_size=128, epochs = epochs, 
            validation_data = (X_val2,Y_val2), verbose=0)
    val_acc_inverted.append(max(history.history['val_acc']))
    print("Run {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(run + 1, epochs, max(history.history['acc']),max(history.history['val_acc']) ))
print("Average validation accuracy: {:.5f}".format(sum(val_acc_inverted) / len(val_acc_inverted)))
from scipy.stats import ttest_ind
ttest_ind(val_acc_normal, val_acc_inverted)
