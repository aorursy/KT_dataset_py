# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import dependencies

import tensorflow as tf

import keras
# setting the paths to the directories

import os

train_path = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train"

val_path = "../input/chest-xray-pneumonia/chest_xray/val"

test_path = "../input/chest-xray-pneumonia/chest_xray/test"

from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Softmax,Input,Flatten,BatchNormalization
model1 = keras.models.Sequential()



model1.add(keras.layers.Flatten(input_shape=(150,150,1)))

model1.add(keras.layers.BatchNormalization())

model1.add(keras.layers.Dense(1024, activation='relu'))

model1.add(keras.layers.Dropout(0.5))

model1.add(keras.layers.Dense(512, activation='relu'))

model1.add(keras.layers.BatchNormalization())

model1.add(keras.layers.Dense(256, activation='relu'))

model1.add(keras.layers.Dense(128, activation='relu'))

model1.add(keras.layers.Dense(2, activation='softmax'))

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.98):

            print("\nReached 98% accuracy so cancelling training!")

            self.model.stop_training = True
gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_batches = gen.flow_from_directory(train_path,target_size=(150,150),color_mode="grayscale",shuffle=True,seed=1,batch_size=16)

callback = myCallback()

model1.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

model1.fit_generator(train_batches, epochs=40, callbacks=[callback])
test_batches = gen.flow_from_directory(test_path,model1.input_shape[1:3],color_mode="grayscale",shuffle=False,seed=1,batch_size=8)

p1 = model1.predict_generator(test_batches, verbose=True)

pre1 = pd.DataFrame(p1)

pre1["filename"] = test_batches.filenames

pre1["label"] = (pre1["filename"].str.contains("PNEUMONIA")).apply(int)

pre1['pre'] = (pre1[1]>0.5).apply(int)

recall_score(pre1["label"],pre1["pre"])
roc_auc_score(pre1["label"],pre1[1])
ans = []

for i in pre1[1]:

    if(i>0.9):

        ans.append(1)

    else:

        ans.append(0)

accuracy_score(pre1["label"], ans)        
from sklearn.metrics import f1_score

f1_score(pre1['label'],ans)
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(pre1["label"], ans)

confusion_matrix
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  

FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)

TP = np.diag(confusion_matrix)

print(FP,FN,TP)

#TN = confusion_matrix.values.sum() - (FP + FN + TP)
cm = confusion_matrix

cm
print("MLP Model Metrics: ")

print("accuracy: ",str(accuracy_score(pre1["label"], ans)))

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

print('Specificity : ', specificity)

print("F1 Score: ", f1_score(pre1['label'],ans))

print("AUC-ROC: ", str(roc_auc_score(pre1["label"],pre1[1])))

precision = cm[0,0]/(cm[0,0]+cm[0,1])

print("Precision: ",str(precision))
model2 = keras.models.Sequential()

model2.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",

                 input_shape=(64,64,1)))

model2.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Dropout(rate=0.25))

model2.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))

model2.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Dropout(rate=0.25))

model2.add(Flatten())

model2.add(Dense(1024,activation="relu"))

model2.add(BatchNormalization())

model2.add(Dropout(rate=0.4))

model2.add(Dense(2, activation="softmax"))
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.98):

            print("\nReached 98% accuracy so cancelling training!")

            self.model.stop_training = True
callback = myCallback()
gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_batches = gen.flow_from_directory(train_path,model2.input_shape[1:3],color_mode="grayscale",shuffle=True,seed=1,batch_size=16)
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'] )

callback = myCallback()

model2.fit_generator(train_batches, epochs=10, callbacks=[callback])
# testing

import pandas as pd

test_batches = gen.flow_from_directory(test_path,model2.input_shape[1:3],color_mode="grayscale",shuffle=False,seed=1,batch_size=8)

p = model2.predict_generator(test_batches, verbose=True)

pre = pd.DataFrame(p)

pre["filename"] = test_batches.filenames

pre["label"] = (pre["filename"].str.contains("PNEUMONIA")).apply(int)

pre['pre'] = (pre[1]>0.5).apply(int)

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,f1_score

recall_score(pre["label"],pre["pre"])
roc_auc_score(pre["label"],pre[1])
ans = []

for i in pre[1]:

    if(i>0.999):

        ans.append(1)

    else:

        ans.append(0)

accuracy_score(pre["label"], ans)        
f1_score(pre['label'],ans)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(pre["label"], ans)

cm
print("CNN Model Metrics: ")

print("accuracy: ",str(accuracy_score(pre["label"], ans)))

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

print('Specificity : ', specificity)

print("F1 Score: ", f1_score(pre['label'],ans))

print("AUC-ROC: ", str(roc_auc_score(pre["label"],pre[1])))

precision = cm[0,0]/(cm[0,0]+cm[0,1])

print("Precision: ",str(precision))
from keras.applications import InceptionResNetV2

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
model3 = keras.models.Sequential()

model3.add(base_model)

model3.add(keras.layers.Flatten())

model3.add(Dense(256,activation='relu'))

model3.add(Dense(2,activation='softmax'))





base_model.trainable = False
gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_batches = gen.flow_from_directory(train_path,model3.input_shape[1:3],batch_size=16)
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'] )

callback = myCallback()

model3.fit_generator(train_batches, epochs=10, callbacks=[callback])
model3.save_weights("incept.h5")
#testing

test_batches = gen.flow_from_directory(test_path,model3.input_shape[1:3],shuffle=False,seed=1,batch_size=8)

p2 = model3.predict_generator(test_batches, verbose=True)

pre2 = pd.DataFrame(p2)

pre2["filename"] = test_batches.filenames

pre2["label"] = (pre2["filename"].str.contains("PNEUMONIA")).apply(int)

pre2['pre'] = (pre2[1]>0.5).apply(int)

recall_score(pre2["label"],pre2["pre"])
roc_auc_score(pre2["label"],pre2[1])
ans = []

for i in pre2[1]:

    if(i>0.5):

        ans.append(1)

    else:

        ans.append(0)

accuracy_score(pre2["label"], ans)        
f1_score(pre2["label"], ans)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(pre2["label"], ans)

cm
print("Transfer Learning Model Metrics: ")

print("accuracy: ",str(accuracy_score(pre2["label"], ans)))

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

print('Specificity : ', specificity)

print("F1 Score: ", f1_score(pre2['label'],ans))

print("AUC-ROC: ", str(roc_auc_score(pre2["label"],pre[1])))

precision = cm[0,0]/(cm[0,0]+cm[0,1])

print("Precision: ",str(precision))