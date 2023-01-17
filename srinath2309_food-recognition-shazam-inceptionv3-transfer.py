# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt; 

# Importing sklearn libraries

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

# Importing Keras libraries

from keras.models import Sequential

from keras.models import Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras import optimizers, losses, activations, models

from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate

from keras import applications

from keras.applications import imagenet_utils

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

 

import warnings

warnings.filterwarnings('ignore')

import cv2

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%cd /kaggle/working/

!ls
!wget https://www.dropbox.com/s/sh5yt160xzqjkk0/Food-11.zip?dl=1
!mv Food-11.zip?dl=1 Food_11.zip
!unzip Food_11.zip
!ls
!rm -rf Food_11.zip
train = [os.path.join("training",img) for img in os.listdir("training")]

val = [os.path.join("validation",img) for img in os.listdir("validation")]

test = [os.path.join("evaluation",img) for img in os.listdir("evaluation")]

len(train),len(val),len(test)
train_y = np.array([int(img.split("/")[-1].split("_")[0]) for img in train])

val_y = np.array([int(img.split("/")[-1].split("_")[0]) for img in val])

test_y = [int(img.split("/")[-1].split("_")[0]) for img in test]

num_classes = 11

# Convert class labels in one hot encoded vector

y_train = []

for x in train_y:

    a = np.array([0]*num_classes)

    a[x] = 1

    y_train.append(a)

y_val = []

for x in val_y:

    a = np.array([0]*num_classes)

    a[x] = 1

    y_val.append(a)

y_test = []

for x in test_y:

    a = np.array([0]*num_classes)

    a[x] = 1

    y_test.append(a)

    

#len(y_train),len(y_val),len(y_test)

y_train = np.array(y_train)

y_val = np.array(y_val)

y_test = np.array(y_test)

y_train.shape,y_val.shape,y_test.shape
# y_train = []

# y_val = []

# y_test = []

# train_y = []

# val_y = []

# test_y = []

# train = []

# val = []

# test = []
import pickle

# with open("test_op.pkl","wb") as file:

#     pickle.dump(y_test,file)

# with open("train_op.pkl","wb") as file:

#     pickle.dump(y_train,file)

# with open("val_op.pkl","wb") as file:

#     pickle.dump(y_val,file)
print("Reading train images..")

X_train = np.array([cv2.resize(cv2.imread(x), dsize=(224,224), interpolation=cv2.INTER_AREA) for x in train])

print("Done.")

X_train.shape
print("Reading val images..")

# outs = []

# X_train = []

X_val = np.array([cv2.resize(cv2.imread(x), dsize=(224,224), interpolation = cv2.INTER_AREA) for x in val])

print("Done.")

X_val.shape
ROWS = 224

COLS = 224

nclass = 11

print(X_train.shape,y_train.shape)

print(X_val.shape,y_val.shape)

checkpointer = ModelCheckpoint(filepath='transfermodel_best.hdf5',

                               verbose=1,save_best_only=True)

base_model = applications.InceptionV3(weights='imagenet', 

                                include_top=False, 

                                input_shape=(ROWS, COLS,3))

base_model.trainable = True

add_model = Sequential()

add_model.add(base_model)

add_model.add(GlobalAveragePooling2D())

add_model.add(Dropout(0.25))

add_model.add(Dense(200, 

                    activation='relu'))

add_model.add(Dense(nclass, 

                    activation='softmax'))

model = add_model

model.compile(loss='categorical_crossentropy', 

              optimizer='adam',

              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=30,

          validation_data=(X_val, y_val), callbacks=[checkpointer],

          verbose=1, shuffle=True)
def plot_acc_loss(history):

    fig = plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)

#     plt.plot(history.history['acc'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

 

    plt.subplot(1, 2, 2)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper right')

    plt.show()

 

plot_acc_loss(history)

model.load_weights("transfermodel_best.hdf5")

preds = np.argmax(model.predict(X_val), axis=1)

print("\nAccuracy of Transfer model (softmax) on validation Data: ", accuracy_score(val_y, preds))

print("\nNumber of correctly identified imgaes: ",

      accuracy_score(val_y, preds, normalize=False),"\n")

confusion_matrix(val_y, preds, labels=range(0,11))
print("Reading test images..")

X_test = np.array([cv2.resize(cv2.imread(x), dsize=(224,224), interpolation = cv2.INTER_AREA) for x in test])

print("Done.")
model.summary()
intermediate_layer_model = Model(inputs=model.input,

                                 outputs=model.get_layer("dense_1").output)

train_features = intermediate_layer_model.predict(np.array(X_train))

# X_train = []

val_features = intermediate_layer_model.predict(np.array(X_val))

# X_val = []

test_features = intermediate_layer_model.predict(np.array(X_test))

# X_test = []
train_features.shape, val_features.shape, test_features.shape
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=0) --> 78.83 val acc

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=170) # --> 78.65

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=180) # --> 78.74

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=190) # --> 78.77

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=200) # --> 78.86

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=210) # --> 78.54

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=220) # --> 78.51

# clf = RandomForestClassifier(max_depth=28,n_estimators=150,random_state=230) # --> 78.89

# clf = RandomForestClassifier(max_depth=28,n_estimators=200,random_state=230) # --> 78.92

# clf = RandomForestClassifier(max_depth=28,n_estimators=220,random_state=230) # --> 78.95

# clf = RandomForestClassifier(max_depth=30,n_estimators=220,random_state=230) # --> 78.97

# clf = RandomForestClassifier(max_depth=45,n_estimators=220,random_state=230) # --> 79.06

clf = RandomForestClassifier(max_depth=45,n_estimators=220,random_state=230)

clf.fit(train_features,np.array(train_y))
from sklearn.svm import SVC

svc = SVC(kernel='rbf',gamma='scale',decision_function_shape='ovo',probability=True)

svc.fit(train_features,np.array(train_y))
SVM_val_outputs = svc.predict(val_features)

SVM_test_outputs = svc.predict(test_features)

SVM_val_outputs.shape, SVM_test_outputs.shape

print("SVM accuracies:")

print("val:",accuracy_score(val_y,SVM_val_outputs))

print("test:",accuracy_score(test_y,SVM_test_outputs))
RF_val_outputs = clf.predict(val_features)

RF_test_outputs = clf.predict(test_features)

RF_val_outputs.shape, RF_test_outputs.shape

print("RF accuracies:")

print("val:",accuracy_score(val_y,RF_val_outputs))

print("test:",accuracy_score(test_y,RF_test_outputs))
model.load_weights("transfermodel_best.hdf5")

tm_val = model.predict(X_val)

tm_test = model.predict(X_test)

preds = np.argmax(tm_val, axis=1)

preds2 = np.argmax(tm_test, axis=1)

print("Transfer Model Accuracies:")

print("val:",accuracy_score(val_y,preds))

print("test:",accuracy_score(test_y,preds2))
SVM_val_outputs = svc.predict_proba(val_features)

SVM_test_outputs = svc.predict_proba(test_features)

SVM_val_outputs.shape, SVM_test_outputs.shape
RF_val_outputs = clf.predict_proba(val_features)

RF_test_outputs = clf.predict_proba(test_features)

RF_val_outputs.shape, RF_test_outputs.shape
# SVM accuracies:

# val: 0.7137026239067056

# test: 0.7367792052584404



# RF accuracies:

# val: 0.7160349854227406

# test: 0.7511204063340304



# Transfer Model Accuracies:

# val: 0.5323615160349854

# test: 0.5602031670152375
w1 = 3.8; w2 = 4.3; w3 = 0# 79

finprobs = []

for i in range(3430):

    p1 = SVM_val_outputs[i].argsort()[-5:][::-1]

    p2 = RF_val_outputs[i].argsort()[-5:][::-1]

    p3 = tm_val[i].argsort()[-5:][::-1]

    p1_scores = sorted(SVM_val_outputs[i])[-5:][::-1]

    p2_scores = sorted(RF_val_outputs[i])[-5:][::-1]

    p3_scores = sorted(tm_val[i])[-5:][::-1]

    probs = [0]*11

    for k in range(5):

        if p1[k]==p2[k] and p1[k] == p3[k]:

            probs[p1[k]] += (w1*p1_scores[k]) + (w2*p2_scores[k]) + (w3*p3_scores[k])

        elif p1[k]==p2[k]:

            probs[p1[k]] += (w1*p1_scores[k]) + (w2*p2_scores[k])

            probs[p3[k]] += (w3*p3_scores[k])

        elif p2[k]==p3[k]:

            probs[p2[k]] += (w2*p2_scores[k]) + (w3*p3_scores[k])

            probs[p1[k]] += (w1*p1_scores[k])

        elif p1[k]==p3[k]:

            probs[p1[k]] += (w1*p1_scores[k]) + (w3*p3_scores[k])

            probs[p2[k]] += (w2*p2_scores[k])

        else:

            probs[p1[k]] += (w1*p1_scores[k])

            probs[p2[k]] += (w2*p2_scores[k])

            probs[p3[k]] += (w3*p3_scores[k])

    probs = np.array(probs).argsort()[-5:][::-1]

    finprobs.append(probs[0])

# print("ensembled!",len(finprobs),len(val_y))

print("val:",accuracy_score(val_y,finprobs))
w1 = 2; w2 = 2; w3 = 1.05# 79

finprobs = []

for i in range(3347):

    p1 = SVM_test_outputs[i].argsort()[-5:][::-1]

    p2 = RF_test_outputs[i].argsort()[-5:][::-1]

    p3 = tm_test[i].argsort()[-5:][::-1]

    p1_scores = sorted(SVM_test_outputs[i])[-5:][::-1]

    p2_scores = sorted(RF_test_outputs[i])[-5:][::-1]

    p3_scores = sorted(tm_test[i])[-5:][::-1]

    probs = [0]*11

    for k in range(5):

        if p1[k]==p2[k] and p1[k] == p3[k]:

            probs[p1[k]] += (w1*p1_scores[k]) + (w2*p2_scores[k]) + (w3*p3_scores[k])

        elif p1[k]==p2[k]:

            probs[p1[k]] += (w1*p1_scores[k]) + (w2*p2_scores[k])

            probs[p3[k]] += (w3*p3_scores[k])

        elif p2[k]==p3[k]:

            probs[p2[k]] += (w2*p2_scores[k]) + (w3*p3_scores[k])

            probs[p1[k]] += (w1*p1_scores[k])

        elif p1[k]==p3[k]:

            probs[p1[k]] += (w1*p1_scores[k]) + (w3*p3_scores[k])

            probs[p2[k]] += (w2*p2_scores[k])

        else:

            probs[p1[k]] += (w1*p1_scores[k])

            probs[p2[k]] += (w2*p2_scores[k])

            probs[p3[k]] += (w3*p3_scores[k])



    probs = np.array(probs).argsort()[-5:][::-1]

    finprobs.append(probs[0])

print("ensembled!",len(finprobs),len(test_y))

print("test:",accuracy_score(test_y,finprobs))