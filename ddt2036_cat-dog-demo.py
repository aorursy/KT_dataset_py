# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os,shutil

src_path="../input"

print(os.listdir(src_path))

#constant value
VALID_SPIT=0.2
IMAGE_SIZE=64
BATCH_SIZE=128
CHANNEL_SIZE=1

# Any results you write to the current directory are saved as output.
label=[]
data=[]
counter=0
path="../input/train/train"
for file in os.listdir(path):
    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_GRAYSCALE)
    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
    try:
        data.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%1000==0:
        print (counter," image data retreived")

data=np.array(data)
data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],1)
label=np.array(label)
print (data.shape)
print (label.shape)
sns.countplot(label)
pd.Series(label).value_counts()
from sklearn.model_selection import train_test_split
train_data, valid_data, train_label, valid_label = train_test_split(
    data, label, test_size=0.2, random_state=42)
print(train_data.shape)
print(train_label.shape)
print(valid_data.shape)
print(valid_label.shape)
sns.countplot(train_label)
pd.Series(train_label).value_counts()
sns.countplot(valid_label)
pd.Series(valid_label).value_counts()
from keras import Sequential
from keras.layers import *
import keras.optimizers as optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
import keras.backend as K
#custom net
model=Sequential()

model.add(Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE)))

model.add(Dense(1024,activation="relu"))

model.add(Dense(128,activation="relu"))

model.add(Dense(1,activation="sigmoid"))

model.summary()
#custom cnn net 
model=Sequential()
model.add(Conv2D(8, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()
# training
model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])

callack_saver = ModelCheckpoint(
            "model.h5"
            , monitor='val_loss'
            , verbose=0
            , save_weights_only=True
            , mode='auto'
            , save_best_only=True
        )

train_history=model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=15,batch_size=BATCH_SIZE, callbacks=[callack_saver])
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'acc', 'val_acc')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Predict the values from the validation dataset
Y_pred = model.predict(valid_data)
predicted_label=np.round(Y_pred,decimals=2)
predicted_label=[1 if value>0.5 else 0 for value in predicted_label]
confusion_mtx = confusion_matrix(valid_label, predicted_label) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2)) 
image_list=[]
test_data=[]
count = 0
for file in os.listdir("../input/test1/test1"):
    image_data=cv2.imread(os.path.join("../input/test1/test1",file))
    image_list.append(image_data)
    
    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_GRAYSCALE)
    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
    test_data.append(image_data/255)
    count +=1
    if count == 1:
        break
        
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(image_list[0])
ax[1].imshow(test_data[0])   
    
test_data=np.array(test_data)
test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],1)
print(test_data.shape)
predicted_labels=model.predict(test_data)
predicted_labels=np.round(predicted_labels,decimals=2)
labels=[1 if value>0.5 else 0 for value in predicted_labels]
print(labels)
layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
f1 = layer_1([test_data])[0]
print(f1.shape)
#第一层卷积后的特征图展示，输出是（1,32,32,8）
for _ in range(8):
        show_img = f1[:, :, :, _]
        show_img.shape = [32, 32]
        plt.subplot(1, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
plt.show()
layer_3 = K.function([model.layers[0].input], [model.layers[3].output])
f1 = layer_3([test_data])[0]#只修改inpu_image
print(f1.shape)
for _ in range(16):
        show_img = f1[:, :, :, _]
        show_img.shape = [16, 16]
        plt.subplot(2, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
plt.show()
layer_5 = K.function([model.layers[0].input], [model.layers[5].output])
f1 = layer_5([test_data])[0]#只修改inpu_image
print(f1.shape)
for _ in range(32):
        show_img = f1[:, :, :, _]
        show_img.shape = [8, 8]
        plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
plt.show()
layer_7 = K.function([model.layers[0].input], [model.layers[7].output])
f1 = layer_7([test_data])[0]#只修改inpu_image
print(f1.shape)
for _ in range(64):
        show_img = f1[:, :, :, _]
        show_img.shape = [4, 4]
        plt.subplot(8, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
plt.show()
test_data=[]
id=[]
counter=0
for file in os.listdir("../input/test1/test1"):
    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_GRAYSCALE)
    try:
        image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
        test_data.append(image_data/255)
        id.append((file.split("."))[0])
    except:
        print ("ek gaya")
    counter+=1
    if counter%1000==0:
        print (counter," image data retreived")

test_data=np.array(test_data)
print (test_data.shape)
test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],1)
dataframe_output=pd.DataFrame({"id":id})
predicted_labels=model.predict(test_data)
predicted_labels=np.round(predicted_labels,decimals=2)
labels=[1 if value>0.5 else 0 for value in predicted_labels]

#print(len(labels))
dataframe_output["label"]=labels
print(dataframe_output)
dataframe_output.to_csv("submission.csv",index=False)