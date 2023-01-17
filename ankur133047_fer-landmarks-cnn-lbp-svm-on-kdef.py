# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

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

train_lan=np.load('../input/kdef-landmarks-lbp/landmarks.npy')
print(train_lan.shape)
valid_lan=np.load('../input/jaffe-landmarks-lbp/landmarks.npy')

print(os.listdir("../input/modified-kdef-facial-expression-dataset"))

# get the data
# filname = '../input/facial-expression/fer2013/fer2013.csv'
filname = '../input/modified-kdef-facial-expression-dataset/kdef_pixels.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']
# df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)
df=pd.read_csv('../input/modified-kdef-facial-expression-dataset/kdef_pixels.csv',names=names, na_filter=False)
im=df['pixels']
df.head(10)

# Read and process train dataset

df = pd.read_csv('../input/modified-kdef-facial-expression-dataset/kdef_pixels.csv')
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
X_train = x_train.reshape(N, 256, 256, 1)
print(X_train.shape, y_train.shape)

num_class = len(set(y_train))
print(num_class)


# Read and process valid dataset

df = pd.read_csv('../input/modified-jaffe-facial-expression-dataset/jaffe_pixels.csv')
df.head()

df["Usage"].value_counts()

public_test_df = df[["emotion", "pixels"]][df["Usage"] == "Training"]
public_test_df.isnull().sum()

public_test_df["pixels"] = public_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_valid = np.vstack(public_test_df["pixels"].values)
y_valid = np.array(public_test_df["emotion"])

x_valid = np.array(x_valid) / 255.0
print(x_valid.shape, y_valid.shape)

N = len(x_valid)
X_valid = x_valid.reshape(N, 256, 256, 1)
print(X_valid.shape, y_valid.shape)

num_class = len(set(y_valid))
print(num_class)



# Read and process test dataset

df = pd.read_csv('../input/modified-ck-facial-expression-dataset/ck_pixels.csv')
df.head()

df["Usage"].value_counts()

private_test_df = df[["emotion", "pixels"]][df["Usage"] == "Training"]
private_test_df.isnull().sum()


private_test_df["pixels"] = private_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_test = np.vstack(private_test_df["pixels"].values)
y_test = np.array(private_test_df["emotion"])
# print(x_test)

x_test = np.array(x_test) / 255.0
# print(x_test)
print(x_test.shape, y_test.shape)

N = len(x_test)
X_test = x_test.reshape(N, 256, 256, 1)
# print(X_test)
print(X_test.shape, y_test.shape)

num_class = len(set(y_test))
print(num_class)



# X_train, X_valid, X_test is used for training CNN

# For applying to LBP+SVM

x_train = x_train.reshape(-1, 256, 256, 1)
x_valid = x_valid.reshape(-1, 256, 256, 1)
x_test = x_test.reshape(-1, 256, 256, 1)

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



print(os.listdir("../input"))
print(os.listdir("../input/kdef-landmarks-lbp"))


# Get landmarks and lbp data

import numpy as np

train_lan=np.load('../input/kdef-landmarks-lbp/landmarks.npy')
valid_lan=np.load('../input/jaffe-landmarks-lbp/landmarks.npy')
test_lan=np.load('../input/ck-landmarks-lbp/landmarks.npy')

train_lbp=np.load('../input/kdef-landmarks-lbp/lbp_features.npy')
valid_lbp=np.load('../input/jaffe-landmarks-lbp/lbp_features.npy')
test_lbp=np.load('../input/ck-landmarks-lbp/lbp_features.npy')

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
print(lbp_valid.shape)
print(lbp_test.shape)




print('LAN+LBP+SVM is going to applicable. resolution doesn\'t')
def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


def matrix(svm_model):

#     validation_accuracy = evaluate(svm_model, lbp_test, y_test)
#     print("  - validation accuracy = {0:.1f}".format(validation_accuracy*100))

#     valid_labels = svm_model.predict(lbp_test)
#     print(classification_report(y_test, valid_labels))
#     mat = confusion_matrix(y_test, valid_labels)
#     print(mat)
    
    
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
    
#     test_accuracy = evaluate(svm_model, lbp_valid, y_test)
#     print("  - test accuracy = {0:.1f}".format(test_accuracy*100))
#     test_labels = svm_model.predict(lbp_valid)
    
#     print(classification_report(y_valid, valid_labels))
#     print(mat)



#     test_accuracy = evaluate(svm_model, lbp_test, y_test)
#     print("  - test accuracy = {0:.1f}".format(test_accuracy*100))

#     test_labels = svm_model.predict(lbp_test)
#     print(classification_report(y_test, test_labels))
#     mat = confusion_matrix(y_test, test_labels)
#     print(mat)





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

def my_model(w,h):
    model = Sequential()
    input_shape = (w,h,1)
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
model=my_model(256,256)
model.summary()


# Training CNN

start_time = time.time()  

path_model='model_filter_2.h5' # save model at this location after each epoch
K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=my_model(256,256) # create the model
K.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model
h=model.fit(x=X_train,     
            y=Y_train, 
            batch_size=64, 
            epochs=20, 
            verbose=1, 
            validation_data=(X_valid,Y_valid),
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
# Application of SVM model 3

start_time = time.time()  
svm_model_3 = LinearSVC(C=100.0)
svm_model_3.fit(feat_train, y_train)
print('fitting done !!!')

matrix2(svm_model_3)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))
# Application of SVM model 4

random_state = 0
epochs = 10000
kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'

start_time = time.time()  
svm_model_4 = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function)
svm_model_4.fit(feat_train, y_train)
print('fitting done !!!')

matrix2(svm_model_4)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))
# use fer2013 as next test case:

# first lbp+svm

test_lan_2 = np.load('../input/train-landmarks-fer2013/PrivateTest_landmarks.npy')

test_lbp_2 = np.load('../input/train-landmarks-fer2013/PrivteTest_lbp_features.npy')

test_lan_2 = np.array([x.flatten() for x in test_lan_2])

print(test_lan_2.shape)
print(test_lbp_2.shape)

lbp_test_2 = np.concatenate((test_lan_2, test_lbp_2), axis=1)
def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy

def matrix3(svm_model, t2, y2):
    
    test_accuracy = evaluate(svm_model, t2, y2)
    print("  - test accuracy = {0:.1f}".format(test_accuracy*100))

    test_labels = svm_model.predict(t2)
    print(classification_report(y2, test_labels))
    mat = confusion_matrix(y2, test_labels)
    print(mat)

# Read and process test dataset

df = pd.read_csv('../input/modified-fer2013/modified_fer2013/fer2013.csv')
df.head()

df["Usage"].value_counts()

private_test_df = df[["emotion", "pixels"]][df["Usage"]=="PrivateTest"]
# private_test_df.isnull().sum()


private_test_df["pixels"] = private_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_test_2 = np.vstack(private_test_df["pixels"].values)
y_test_2 = np.array(private_test_df["emotion"])
# print(x_test)

x_test_2 = np.array(x_test_2) / 255.0
# print(x_test)
print(x_test_2.shape, y_test_2.shape)

N = len(x_test_2)
X_test_2 = x_test_2.reshape(N, 48, 48, 1)
# print(X_test)
print(X_test_2.shape, y_test_2.shape)

Y_test_2 = (np.arange(num_class) == y_test_2[:, None]).astype(np.float32)

num_class = len(set(y_test_2))
print(num_class)
# Application of SVM model by fer

start_time = time.time()  
matrix3(svm_model_1, lbp_test_2, y_test_2)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))
# Application of SVM model by fer

start_time = time.time()  
matrix3(svm_model_2, lbp_test_2, y_test_2)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))
# For CNN+SVM

model=my_model(48,48)
model.summary()
# Training CNN

start_time = time.time()  

path_model='model_filter_2.h5' # save model at this location after each epoch
K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=my_model(48,48) # create the model
K.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model
h=model.fit(x=X_test_2,     
            y=Y_test_2, 
            batch_size=64, 
            epochs=20, 
            verbose=1, 
#             validation_data=(X_valid,Y_valid),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )

training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))
# Find earlier layer value of CNN anduse as feature extractor

model_feat_2 = Model(inputs=model.input,outputs=model.get_layer('dense_one').output)

print(model.input, model.get_layer('dense_one').output)

feat_test_2 = model_feat_2.predict(X_test_2)
feat_test_2 = np.concatenate((test_lan_2, feat_test_2), axis=1)
print(feat_test_2.shape)
# Application of SVM model by fer

start_time = time.time()  
matrix3(svm_model_3, feat_test_2, y_test_2)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))
# Application of SVM model by fer

start_time = time.time()  
matrix3(svm_model_4, feat_test_2, y_test_2)
training_time = time.time() - start_time
print("training time = {0:.1f} sec".format(training_time))

