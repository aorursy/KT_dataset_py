# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt; 

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

# Importing sklearn libraries

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

# Importing Keras libraries

from keras.utils import np_utils

from keras.models import Sequential

from keras.applications import VGG16

from keras.applications import imagenet_utils

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.layers import Dense, Conv2D, MaxPooling2D

from keras.layers import Dropout, Flatten, GlobalAveragePooling2D

 

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

# len(test)
train_y = [int(img.split("/")[-1].split("_")[0]) for img in train]

val_y = [int(img.split("/")[-1].split("_")[0]) for img in val]

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
import pickle

with open("test_op.pkl","wb") as file:

    pickle.dump(y_test,file)

with open("train_op.pkl","wb") as file:

    pickle.dump(y_train,file)

with open("val_op.pkl","wb") as file:

    pickle.dump(y_val,file)
print("Reading train images..")

X_train = [cv2.resize(cv2.imread(x), dsize=(224,224), interpolation=cv2.INTER_AREA) for x in train]

print("Done.")

len(X_train)
model = VGG16(weights="imagenet", include_top=False)

outs = []

for img in X_train:

    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    outs.append(model.predict(x)[0])

    if len(outs) % 100 == 0:

        print(len(outs))

outs = np.array(outs)

outs.shape

# print("creating train features..")

# train_x, train_features, train_features_flatten = create_features(train, model)

# print("creating val features..")

# val_x, val_features, val_features_flatten = create_features(val, model)

# test_x, test_features, test_features_flatten = create_features(test, model)



# print(train_x.shape, train_features.shape, train_features_flatten.shape)

# print(val_x.shape, val_features.shape, val_features_flatten.shape)

# print(test_x.shape, test_features.shape, test_features_flatten.shape)
print(outs.shape)

with open("train_features.pkl","wb") as file:

    pickle.dump(outs,file)
print("Reading val images..")

outs = []

X_train = []

X_val = [cv2.resize(cv2.imread(x), dsize=(224,224), interpolation = cv2.INTER_AREA) for x in val]

print("Done.")

len(X_val)
model = VGG16(weights="imagenet", include_top=False)

for img in X_val:

    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    outs.append(model.predict(x)[0])

    if len(outs) % 100 == 0:

        print(len(outs))

outs = np.array(outs)

outs.shape
print(outs.shape)

with open("val_features.pkl","wb") as file:

    pickle.dump(outs,file)
outs = []

X_val = []

print("Reading test images..")

X_test = [cv2.resize(cv2.imread(x), dsize=(224,224), interpolation = cv2.INTER_AREA) for x in test]

print("Done.")

len(X_test)
model = VGG16(weights="imagenet", include_top=False)

for img in X_test:

    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    outs.append(model.predict(x)[0])

    if len(outs) % 100 == 0:

        print(len(outs))

outs = np.array(outs)

outs.shape
print(outs.shape)

with open("test_features.pkl","wb") as file:

    pickle.dump(outs,file)
outs = []

X_train = []

X_val = []

X_test = []



# val_features = []

# train_features = []

val_features = pickle.load(open("val_features.pkl","rb"))

train_features = pickle.load(open("train_features.pkl","rb"))

test_features = pickle.load(open("test_features.pkl","rb"))
y_test = []

with open("test_op.pkl","rb") as file:

    y_test = pickle.load(file)

y_train = []

with open("train_op.pkl","rb") as file:

    y_train = pickle.load(file)

y_val = []

with open("val_op.pkl","rb") as file:

    y_val = pickle.load(file)
print(train_features.shape, y_train.shape)

print(val_features.shape, y_val.shape)

print(test_features.shape,y_test.shape)
checkpointer = ModelCheckpoint(filepath='transfermodel_best.hdf5',

                               verbose=1,save_best_only=True)

model_transfer = Sequential()

model_transfer.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))

model_transfer.add(Dropout(0.2))

model_transfer.add(Dense(120, activation='relu'))

model_transfer.add(Dense(256, activation='relu'))

model_transfer.add(Dense(11, activation='softmax'))

model_transfer.compile(loss='categorical_crossentropy', optimizer='adam',

              metrics=['accuracy'])

history = model_transfer.fit(train_features, y_train, batch_size=64, epochs=30,

          validation_data=(val_features, y_val), callbacks=[checkpointer],

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

model_transfer.summary()
from keras.models import Model

intermediate_layer_model = Model(inputs=model_transfer.input,

                                 outputs=model_transfer.get_layer("dense_2").output)

train_feats = intermediate_layer_model.predict(train_features)

train_features = []

val_feats = intermediate_layer_model.predict(val_features)

# val_features = []

test_feats = intermediate_layer_model.predict(test_features)

# test_features = []
train_feats.shape,val_feats.shape,test_feats.shape
len(train_y), len(val_y), len(test_y)
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

clf.fit(train_feats,np.array(train_y))
from sklearn.svm import SVC

svc = SVC(kernel='rbf',gamma='scale',decision_function_shape='ovo',probability=True)

svc.fit(train_feats,np.array(train_y))
SVM_val_outputs = svc.predict(val_feats)

SVM_test_outputs = svc.predict(test_feats)

SVM_val_outputs.shape, SVM_test_outputs.shape

print("SVM accuracies:")

print("val:",accuracy_score(val_y,SVM_val_outputs))

print("test:",accuracy_score(test_y,SVM_test_outputs))
RF_val_outputs = clf.predict(val_feats)

RF_test_outputs = clf.predict(test_feats)

RF_val_outputs.shape, RF_test_outputs.shape

print("RF accuracies:")

print("val:",accuracy_score(val_y,RF_val_outputs))

print("test:",accuracy_score(test_y,RF_test_outputs))
tm_val = model_transfer.predict(val_features)

tm_test = model_transfer.predict(test_features)

preds = np.argmax(tm_val, axis=1)

print("Transfer model accuracies:")

print("val", accuracy_score(val_y, preds))

preds2 = np.argmax(tm_test, axis=1)

print("test", accuracy_score(test_y, preds2))
SVM_val_outputs = svc.predict_proba(val_feats)

SVM_test_outputs = svc.predict_proba(test_feats)

SVM_val_outputs.shape, SVM_test_outputs.shape
RF_val_outputs = clf.predict_proba(val_feats)

RF_test_outputs = clf.predict_proba(test_feats)

RF_val_outputs.shape, RF_test_outputs.shape
len(val_y)
# SVM accuracies:

# val: 0.7880466472303207

# test: 0.8037048102778608

# RF accuracies:

# val: 0.79067055393586

# test: 0.8099790857484315

# Transfer model accuracies:

# val 0.7740524781341108

# test 0.7842844338213325

# w1 = 2; w2 = 2; w3 = 1.05 --> 79.09 val acc.
w1 = 2; w2 = 2; w3 = 1.05# 79

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
print(len(test_features))

preds = np.argmax(model_transfer.predict(test_features), axis=1)

print("\nAccuracy on Test Data: ", accuracy_score(test_y, preds))

print("\nNumber of correctly identified imgaes: ",

      accuracy_score(test_y, preds, normalize=False),"\n")

confusion_matrix(test_y, preds, labels=range(0,11))
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

print("val:",accuracy_score(test_y,finprobs))