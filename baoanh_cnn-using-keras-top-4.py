# import all of library used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split, KFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.utils import np_utils  # fonctionality like get_dummies of pandas
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('tf') # to change the shape of image is (samples, width, height, channels)
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau # changing learing rate while training data
from keras.layers.normalization import BatchNormalization


# load data from keras
from keras.datasets import mnist

%matplotlib inline
plt.rcParams['figure.figsize'] = [16, 8]
import os
os.listdir("../input/")
# load data set
train_kaggle = pd.read_csv("../input/train.csv")
test_kaggle = pd.read_csv("../input/test.csv")
print("train shape: ", train_kaggle.shape)
print("test shape: ", test_kaggle.shape)
train_target_kaggle = train_kaggle.iloc[:, :1].values
train_kaggle = train_kaggle.iloc[:,1:].values.astype('float32')
test_kaggle = test_kaggle.iloc[:,:].values.astype('float32')
# standalize data
# this ways is normally used in the way of model of machine learning, like LightGBM, SVM...
'''
scale = np.max(train_kaggle)
train_kaggle /= scale
test_kaggle /= scale

mean = np.std(train_kaggle)
train_kaggle -= mean
test_kaggle -= mean
'''
# in the case used by CNN, we normalize data by divide 255 (color)
train_kaggle = train_kaggle/255
test_kaggle = test_kaggle/255
# These 2 ways below to OneHotEncoding of the label that have multiple label, in this case: 0-9 (10 categories)
#target = np_utils.to_categorical(target)
print("Before:", train_target_kaggle.shape)
train_target_kaggle = pd.get_dummies(train_target_kaggle.ravel())
print("After:", train_target_kaggle.shape)
# split data into training set and testing set
X_train, X_val, y_train, y_val = train_test_split(train_kaggle, train_target_kaggle, test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape
# a function to create input layer, hidden layer and output layer and then fully connected
def baseline_model():
    model = Sequential()
    model.add(Dense(784,input_dim=784, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
    return model
# instance model
model = baseline_model()
model.summary()
# instance history to record all of variable of results
epochs = 35
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=len(X_train)//epochs, verbose=2)
# show the model performance
scores = model.evaluate(X_val, y_val, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# plot the accuracy and loss in each process: training and validation
def plot_(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    f, [ax1, ax2] = plt.subplots(1,2)
    ax1.plot(range(len(acc)), acc, label="acc")
    ax1.plot(range(len(acc)), val_acc, label="val_acc")
    ax1.set_title("Training Accuracy vs Validation Accuracy")
    ax1.legend()

    ax2.plot(range(len(loss)), loss, label="loss")
    ax2.plot(range(len(loss)), val_loss, label="val_loss")
    ax2.set_title("Training Loss vs Validation Loss")
    ax2.legend()

plot_(history)
# rescale the data
X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)

X_train.shape, X_val.shape

def model_baseline_CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(14, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(14, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
    return model
modelCNN = model_baseline_CNN()
modelCNN.summary()
history = modelCNN.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=len(X_train)//epochs, verbose=2)
# plot the accuracy and loss in each process: training and validation  
plot_(history)
score = modelCNN.evaluate(X_val, y_val, verbose=0)
print("Baseline error %.2f%%" % (100-score[1]*100))
# save kaggle dataset
X_train_kaggle, X_val_kaggle, y_train_kaggle, y_val_kaggle = X_train, X_val, y_train, y_val
X_train_kaggle.shape, X_val_kaggle.shape, y_train_kaggle.shape, y_val_kaggle.shape
# load data from keras dataset
(X_train_keras, y_train_keras), (X_val_keras, y_val_keras) = mnist.load_data()
X_train_keras.shape, y_train_keras.shape, X_val_keras.shape, y_val_keras.shape
# rescale data
X_train_keras = X_train_keras.reshape(-1,28,28,1).astype('float32')
X_val_keras = X_val_keras.reshape(-1,28,28,1).astype('float32')
y_train_keras = pd.get_dummies(y_train_keras)
y_val_keras = pd.get_dummies(y_val_keras)
X_train_keras.shape, X_val_keras.shape, y_train_keras.shape, y_val_keras.shape
#normalize data
X_train_keras /= 255
X_val_keras /= 255
X_train = np.concatenate((X_train_kaggle, X_train_keras), axis=0)
X_val = np.concatenate((X_val_kaggle, X_val_keras), axis=0)
y_train = np.concatenate((y_train_kaggle, y_train_keras), axis=0)
y_val = np.concatenate((y_val_kaggle, y_val_keras), axis=0)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
modelCNN = model_baseline_CNN()
modelCNN.summary()
# we inscrease epochs and batch size to fit all of data
history = modelCNN.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=len(X_train)//epochs, verbose=2)
plot_(history)
score = modelCNN.evaluate(X_val, y_val, verbose=0)
print("Baseline error %.2f%%" % (100-score[1]*100))
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
modelCNN = model_baseline_CNN()
history = modelCNN.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=len(X_train)//epochs, 
                       verbose=2, callbacks=[learning_rate_reduction])
plot_(history)
score = modelCNN.evaluate(X_val, y_val, verbose=0)
print("Baseline error %.2f%%" % (100-score[1]*100))
# reshape the test set
test_kaggle = test_kaggle.reshape(-1,28,28,1).astype('float32')
pred_target = modelCNN.predict(test_kaggle)
pred_target = np.argmax(pred_target, axis=1)
submit = pd.read_csv("../input/sample_submission.csv")
submit['Label'] = pred_target
submit.to_csv("results.csv", index=False)


