# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/facial-expression"))
print(os.listdir("../input/modified-fer2013"))
print(os.listdir("../input/train-landmarks-fer2013"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_lan=np.load('../input/train-landmarks-fer2013/training_landmarks.npy')
print(train_lan.shape)
valid_lan=np.load('../input/train-landmarks-fer2013/PublicTestlandmarks.npy')

# get the data
# filname = '../input/facial-expression/fer2013/fer2013.csv'
filname = '../input/modified-fer2013/modified_fer2013/fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']
# df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)
df=pd.read_csv('../input/modified-fer2013/modified_fer2013/fer2013.csv',names=names, na_filter=False)
im=df['pixels']
df.head(10)

def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

X, Y = getData(filname)
num_class = len(set(Y))
print(num_class)
# keras with tensorflow backend
# N, D = X.shape
N = len(X)
print(N)
X = X.reshape(N, 48, 48, 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
def my_model():
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ############
#     model.add(Flatten())
#     model.add(Dense(256, name="dense_one"))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
    ###########
    
    model.add(Flatten())
    model.add(Dense(128, name="dense_one"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    #model.summary()
    
    return model
model=my_model()
model.summary()
path_model='model_filter_2.h5' # save model at this location after each epoch
K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=my_model() # create the model
K.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model
h=model.fit(x=X_train,     
            y=y_train, 
            batch_size=64, 
            epochs=20, 
            verbose=1, 
            validation_data=(X_test,y_test),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )
from keras.utils import np_utils

# df = pd.read_csv('../input/facial-expression/fer2013/fer2013.csv')
df = pd.read_csv('../input/modified-fer2013/modified_fer2013/fer2013.csv')


df.head()

df["Usage"].value_counts()

train = df[["emotion", "pixels"]][df["Usage"] == "Training"]
train.isnull().sum()


train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_train = np.vstack(train['pixels'].values)
y_train = np.array(train["emotion"])
print(x_train.shape, y_train.shape)



public_test_df = df[["emotion", "pixels"]][df["Usage"]=="PublicTest"]

public_test_df["pixels"] = public_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_valid = np.vstack(public_test_df["pixels"].values)
y_valid = np.array(public_test_df["emotion"])


private_test_df = df[["emotion", "pixels"]][df["Usage"]=="PrivateTest"]

private_test_df["pixels"] = private_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_test = np.vstack(public_test_df["pixels"].values)
y_test = np.array(public_test_df["emotion"])



x_train = x_train.reshape(-1, 48, 48, 1)
x_valid = x_valid.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)
print(x_train.shape, x_valid.shape)

print(y_train)

# y_train = np_utils.to_categorical(y_train)
# y_valid = np_utils.to_categorical(y_valid)
# y_test = np_utils.to_categorical(y_test)
print(y_train.shape, y_valid.shape)
import numpy as np
from keras.models import load_model, Model
from sklearn.svm import LinearSVC, SVC, NuSVC
train_lan=np.load('../input/train-landmarks-fer2013/training_landmarks.npy')
valid_lan=np.load('../input/train-landmarks-fer2013/PublicTestlandmarks.npy')
test_lan=np.load('../input/train-landmarks-fer2013/PrivateTest_landmarks.npy')

train_lbp=np.load('../input/train-landmarks-fer2013/training_lbp_features.npy')
valid_lbp=np.load('../input/train-landmarks-fer2013/PublicTest_lbp_features.npy')
test_lbp=np.load('../input/train-landmarks-fer2013/PrivteTest_lbp_features.npy')

print(train_lbp.shape)

train_lan = np.array([x.flatten() for x in train_lan])
valid_lan = np.array([x.flatten() for x in valid_lan])
test_lan = np.array([x.flatten() for x in test_lan])
print(train_lan.shape)


train_lan = np.concatenate((train_lan, train_lbp), axis=1)
valid_lan = np.concatenate((valid_lan, valid_lbp), axis=1)
test_lan = np.concatenate((test_lan, test_lbp), axis=1)
print(train_lan.shape)


model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_one').output)

print(model.input, model.get_layer('dense_one').output)

feat_train = model_feat.predict(x_train)
print(feat_train.shape)
feat_train = np.concatenate((feat_train, train_lan), axis=1)
print(feat_train.shape)


feat_val = model_feat.predict(x_valid)
feat_val = np.concatenate((feat_val, valid_lan), axis=1)
print(feat_val.shape)


feat_test = model_feat.predict(x_test)
feat_test = np.concatenate((feat_test, test_lan), axis=1)
print(feat_test.shape)

svm_model = LinearSVC(C=100.0)


svm_model.fit(feat_train, y_train)

print('fitting done !!!')

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy

    
validation_accuracy = evaluate(svm_model, feat_val, y_valid)
print("  - validation accuracy = {0:.1f}".format(validation_accuracy*100))

valid_labels = svm_model.predict(feat_val)
print(classification_report(y_valid, valid_labels))
mat = confusion_matrix(y_valid, valid_labels)
print(mat)

test_accuracy = evaluate(svm_model, feat_test, y_test)
print("  - test accuracy = {0:.1f}".format(test_accuracy*100))

test_labels = svm_model.predict(feat_test)
print(classification_report(y_test, test_labels))
mat = confusion_matrix(y_test, test_labels)
print(mat)



