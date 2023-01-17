# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!wget https://www.dropbox.com/s/sh5yt160xzqjkk0/Food-11.zip?dl=1
!mv Food-11.zip?dl=1 Food_11.zip
!unzip Food_11.zip
!rm -rf Food_11.zip
train = [os.path.join("training",img) for img in os.listdir("training")]
val = [os.path.join("validation",img) for img in os.listdir("validation")]
test = [os.path.join("evaluation",img) for img in os.listdir("evaluation")]
len(train),len(val),len(test)
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
# print("Reading train images..")
# X_train = [cv2.resize(cv2.imread(x), dsize=(224,224), interpolation=cv2.INTER_AREA) for x in train]
print("Reading val images..")
X_val = [cv2.resize(cv2.imread(x), dsize=(224,224), interpolation = cv2.INTER_AREA) for x in val]
print("Done.")
# len(X_train), len(X_val)
# X_train = np.array(X_train)
X_val = np.array(X_val)
# X_train.shape, X_val.shape
import matplotlib.pyplot as plt
def plot_acc_loss(history):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
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
# !pip install --upgrade pip setuptools wheel
# !pip install -I tensorflow
!pip install -I keras

import tensorflow
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.layers import Flatten, Input, Dense

checkpointer = ModelCheckpoint(filepath='resnet50_trainabletrue.hdf5',
                               verbose=1,save_best_only=True)

base_model = ResNet50(weights='imagenet', include_top = False,input_shape=(224,224,3))
base_model.trainable = False
model_transfer = Sequential()
model_transfer.add(base_model)
model_transfer.add(GlobalAveragePooling2D())
model_transfer.add(Dropout(0.2))
model_transfer.add(Dense(100, activation='relu'))
model_transfer.add(Dense(11, activation='softmax'))
model_transfer.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model_transfer.fit(X_train, y_train, batch_size=32, epochs=10,
          validation_data=(X_val, y_val), callbacks=[checkpointer],
          verbose=1, shuffle=True)

plot_acc_loss(history)
print("Reading test images..")
X_test = [cv2.resize(cv2.imread(x), dsize=(224,224), interpolation = cv2.INTER_AREA) for x in test]
X_test = np.array(X_test)
X_test.shape
model_transfer.summary()
from tensorflow.keras.models import Model
intermediate_layer_model = Model(inputs=model_transfer.input,
                                 outputs=model_transfer.get_layer("dense_4").output)
train_feats = intermediate_layer_model.predict(X_train)
del X_train
val_feats = intermediate_layer_model.predict(X_val)
del X_val
test_feats = intermediate_layer_model.predict(X_test)
del X_test
train_feats.shape,val_feats.shape,test_feats.shape
len(train_y), len(val_y), len(test_y)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=40,n_estimators=150,random_state=100)
clf.fit(train_feats,np.array(train_y))
RF_val_outputs = clf.predict(val_feats)
RF_test_outputs = clf.predict(test_feats)
RF_val_outputs.shape, RF_test_outputs.shape
print("RF accuracies:")
print("val:",accuracy_score(val_y,RF_val_outputs))
print("test:",accuracy_score(test_y,RF_test_outputs))
from sklearn.svm import SVC
svc = SVC(kernel='rbf',gamma='scale',decision_function_shape='ovo',probability=True)
svc.fit(train_feats,np.array(train_y))
SVM_val_outputs = svc.predict(val_feats)
SVM_test_outputs = svc.predict(test_feats)
SVM_val_outputs.shape, SVM_test_outputs.shape
print("SVM accuracies:")
print("val:",accuracy_score(val_y,SVM_val_outputs))
print("test:",accuracy_score(test_y,SVM_test_outputs))
model_transfer = Sequential()
model_transfer.add(base_model)
model_transfer.add(GlobalAveragePooling2D())
model_transfer.add(Dropout(0.2))
model_transfer.add(Dense(100, activation='relu'))
model_transfer.add(Dense(11, activation='softmax'))
model_transfer.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model_transfer.load_weights("resnet50_trainabletrue.hdf5")
tm_val = model_transfer.predict(X_val)
tm_test = model_transfer.predict(X_test)
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
# w1 = 3.5; w2 = 1.8; w3 = 1.05# 79
# # xception 
#     # voting scheme: 
#         val: 0.8495626822157435
#         test: 0.8780997908574844
#     # weighted scheme:
#         val: 0.8580174927113703
#         test: 0.8837765162832387
w1 = 4; w2 = 2; w3 = 0.5# 79
# w1 = 1; w2 = 1; w3 = 1# 79
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
print(0.7317,"- to beat.")
print("val:",accuracy_score(val_y,finprobs))
w1 = 4; w2 = 2; w3 = 0.5# 79
# w1 = 1; w2 = 1; w3 = 1# 79
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
print(0.7597,"- to beat.")
print("test:",accuracy_score(test_y,finprobs))

from keras.applications.xception import Xception
from keras.models import Sequential
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
# from keras.applications.xception import preprocess_input, decode_predictions
from keras.layers import Flatten, Input, Dense

checkpointer = ModelCheckpoint(filepath='xception.hdf5',
                               verbose=1,save_best_only=True)

base_model = Xception(weights='imagenet', include_top = False,input_shape=(224,224,3))
base_model.trainable = True
model_transfer = Sequential()
model_transfer.add(base_model)
model_transfer.add(GlobalAveragePooling2D())
model_transfer.add(Dropout(0.2))
model_transfer.add(Dense(100, activation='relu'))
model_transfer.add(Dense(11, activation='softmax'))
model_transfer.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model_transfer.fit(X_train, y_train, batch_size=64, epochs=10,
          validation_data=(X_val, y_val), callbacks=[checkpointer],
          verbose=1, shuffle=True)

model_transfer.summary()
predprobs = model_transfer.predict(X_test)
ytrue = np.array([np.argmax(x) for x in y_test])
ypred = []
for pred in predprobs:
    ypred.append(np.argmax(pred))
from sklearn.metrics import accuracy_score
ypred = np.array(ypred)
accuracy_score(ytrue,ypred)
