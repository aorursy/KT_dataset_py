# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/facial-expression"))
print(os.listdir("../input/modified-fer2013"))
print(os.listdir("../input/train-landmarks-fer2013"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_lan=np.load('../input/train-landmarks-fer2013/training_landmarks.npy')
print(train_lan.shape)
train_lbp=np.load('../input/train-landmarks-fer2013/training_lbp_features.npy')
print(train_lbp.shape)

# get the data

filname = '../input/modified-fer2013/modified_fer2013/fer2013.csv'
label_map = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
names=['emotion','pixels','usage']

df=pd.read_csv('../input/modified-fer2013/modified_fer2013/fer2013.csv',names=names, na_filter=False)
im=df['pixels']
df.head(10)

# Read and process train dataset

df = pd.read_csv('../input/modified-fer2013/modified_fer2013/fer2013.csv')
df.head()

df["Usage"].value_counts()

train = df[["emotion", "pixels"]][df["Usage"] == "Training"]
train.isnull().sum()


train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_train = np.vstack(train['pixels'].values)
y_train = np.array(train["emotion"])
print(x_train.shape, y_train.shape)

x_train = np.array(x_train) / 255.0
print(x_train.shape, y_train.shape)

N = len(x_train)
X_train = x_train.reshape(N, 48, 48, 1)
print(X_train.shape, y_train.shape)

num_class = len(set(y_train))
print(num_class)
# Read and process valid dataset

public_test_df = df[["emotion", "pixels"]][df["Usage"]=="PublicTest"]

public_test_df["pixels"] = public_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_valid = np.vstack(public_test_df["pixels"].values)
y_valid = np.array(public_test_df["emotion"])

x_valid = np.array(x_valid) / 255.0
print(x_valid.shape, y_valid.shape)

N = len(x_valid)
X_valid = x_valid.reshape(N, 48, 48, 1)
print(X_valid.shape, y_valid.shape)

num_class = len(set(y_valid))
print(num_class)
# Read and process test dataset

private_test_df = df[["emotion", "pixels"]][df["Usage"]=="PrivateTest"]

private_test_df["pixels"] = private_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_test = np.vstack(public_test_df["pixels"].values)
y_test = np.array(public_test_df["emotion"])
print(x_test)

x_test = np.array(x_test) / 255.0
print(x_test)
print(x_test.shape, y_test.shape)

N = len(x_test)
X_test = x_test.reshape(N, 48, 48, 1)
print(X_test)
print(X_test.shape, y_test.shape)

num_class = len(set(y_test))
print(num_class)
# X_train, X_valid, X_test is used for training CNN

# For applying to LBP+SVM

x_train = x_train.reshape(-1, 48, 48, 1)
x_valid = x_valid.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)

print(x_train.shape, x_valid.shape, x_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)
print(y_test)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Process y for CNN (get capital Y)

Y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
Y_valid = (np.arange(num_class) == y_valid[:, None]).astype(np.float32)
Y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
print(Y_train.shape, Y_valid.shape, Y_test.shape)
print(Y_test)
# Get landmarks and lbp data

import numpy as np

train_lan=np.load('../input/train-landmarks-fer2013/training_landmarks.npy')
valid_lan=np.load('../input/train-landmarks-fer2013/PublicTestlandmarks.npy')
test_lan=np.load('../input/train-landmarks-fer2013/PrivateTest_landmarks.npy')

train_lbp=np.load('../input/train-landmarks-fer2013/training_lbp_features.npy')
valid_lbp=np.load('../input/train-landmarks-fer2013/PublicTest_lbp_features.npy')
test_lbp=np.load('../input/train-landmarks-fer2013/PrivteTest_lbp_features.npy')

print(train_lan.shape)
print(train_lbp.shape)

train_lan = np.array([x.flatten() for x in train_lan])
valid_lan = np.array([x.flatten() for x in valid_lan])
test_lan = np.array([x.flatten() for x in test_lan])
print(train_lan.shape)


lbp_train = np.concatenate((train_lan, train_lbp), axis=1)
lbp_valid = np.concatenate((valid_lan, valid_lbp), axis=1)
lbp_test = np.concatenate((test_lan, test_lbp), axis=1)
print(lbp_train.shape)
def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


def matrix(svm_model):

    validation_accuracy = evaluate(svm_model, lbp_valid, y_valid)
    print("  - validation accuracy = {0:.1f}".format(validation_accuracy*100))

    valid_labels = svm_model.predict(lbp_valid)
    print(classification_report(y_valid, valid_labels))
    mat = confusion_matrix(y_valid, valid_labels)
    print(mat)

    test_accuracy = evaluate(svm_model, lbp_test, y_test)
    print("  - test accuracy = {0:.1f}".format(test_accuracy*100))

    test_labels = svm_model.predict(lbp_test)
    print(classification_report(y_test, test_labels))
    mat = confusion_matrix(y_test, test_labels)
    print(mat)
import time
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
# Application of SVM model 1

start_time = time.time()
svm_model_1 = LinearSVC(C=100.0)
svm_model_1.fit(lbp_train, y_train)
print('fitting done !!!')

matrix(svm_model_1)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))
# Application of SVM model 2

random_state = 0
epochs = 10000
kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'

start_time = time.time()    
svm_model_2 = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function)
svm_model_2.fit(lbp_train, y_train)
print('fitting done !!!')

matrix(svm_model_2)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))

# Landmark+LBP+SVM is done

# Landmark+CNN+SVM is starting
import tensorflow as tf
import keras
import time

from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_json, load_model, Model

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
# CNN model declaration

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
# Training CNN

start_time = time.time()  

path_model='model_filter_2.h5' # save model at this location after each epoch
K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=my_model() # create the model
K.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model
h=model.fit(x=X_train,     
            y=Y_train, 
            batch_size=64, 
            epochs=20, 
            verbose=1, 
            validation_data=(X_test,Y_test),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )

training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))

# Find earlier layer value of CNN anduse as feature extractor

model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_one').output)

print(model.input, model.get_layer('dense_one').output)

feat_train = model_feat.predict(X_train)
print(feat_train.shape)
feat_train = np.concatenate((train_lan, feat_train), axis=1)
print(feat_train.shape)

feat_valid = model_feat.predict(X_valid)
feat_valid = np.concatenate((valid_lan, feat_valid), axis=1)
print(feat_valid.shape)

feat_test = model_feat.predict(X_test)
feat_test = np.concatenate((test_lan, feat_test), axis=1)
print(feat_test.shape)

def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy

def matrix2(svm_model):

    validation_accuracy = evaluate(svm_model, feat_valid, y_valid)
    print("  - validation accuracy = {0:.1f}".format(validation_accuracy*100))

    valid_labels = svm_model.predict(feat_valid)
    print(classification_report(y_valid, valid_labels))
    mat = confusion_matrix(y_valid, valid_labels)
    print(mat)

    test_accuracy = evaluate(svm_model, feat_test, y_test)
    print("  - test accuracy = {0:.1f}".format(test_accuracy*100))

    test_labels = svm_model.predict(feat_test)
    print(classification_report(y_test, test_labels))
    mat = confusion_matrix(y_test, test_labels)
    print(mat)
# Application of SVM model 1

start_time = time.time()  
svm_model_1 = LinearSVC(C=100.0)
svm_model_1.fit(feat_train, y_train)
print('fitting done !!!')

matrix2(svm_model_1)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))

# Application of SVM model 2

random_state = 0
epochs = 10000
kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'

start_time = time.time()  
svm_model_2 = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function)
svm_model_2.fit(feat_train, y_train)
print('fitting done !!!')

matrix2(svm_model_2)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))

