# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set_style({'xtick.bottom':False,
               'ytick.left':False,
               'axes.spines.bottom': False,
               'axes.spines.left': False,
               'axes.spines.right': False,
               'axes.spines.top': False})
files = []
for dirname, _, filenames in os.walk('../input/intel-image-classification/seg_test'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
images = random.sample(files,k=9)
fig = plt.figure(figsize=(15,15))
for i,v in enumerate(images):
    img = plt.imread(v)
    ax = fig.add_subplot(3,3,i+1)
    ax.set_title(v.split('/')[-2])
    ax.imshow(img)
train_datagen = ImageDataGenerator(rescale=1./255,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory("../input/intel-image-classification/seg_train/seg_train",
                                              target_size=(128,128),
                                              batch_size=32,
                                              class_mode="categorical")

test_data = test_datagen.flow_from_directory("../input/intel-image-classification/seg_test/seg_test",
                                              target_size=(128,128),
                                              batch_size=32,
                                              class_mode="categorical")
from tensorflow.keras.optimizers import Adam,SGD,RMSprop

accuracies_ = []

def train(optimizer,train_data,test_data,name,epochs=30):    
    classifier = keras.Sequential([keras.layers.Conv2D(16,(3,3),input_shape=(128,128,3),activation='relu'),
                                  keras.layers.MaxPool2D(2,2),
                                  keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  keras.layers.MaxPool2D(2,2),
                                  keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  keras.layers.MaxPool2D(2,2),
                                  keras.layers.Conv2D(128,(3,3),activation='relu'),
                                  keras.layers.MaxPool2D(2,2),
                                  keras.layers.Conv2D(256,(3,3),activation='relu'),
                                  keras.layers.MaxPool2D(2,2),

                                  keras.layers.Flatten(),
                                  keras.layers.Dense(2048,activation='relu'),
                                  keras.layers.Dropout(0.2),
                                  keras.layers.Dense(1024,activation='relu'),
                                  keras.layers.Dropout(0.1),
                                  keras.layers.Dense(512,activation='relu'),
                                  keras.layers.Dropout(0.1),
                                  keras.layers.Dense(256,activation='relu'),
                                  keras.layers.Dense(128,activation='relu'),
                                  keras.layers.Dense(64,activation='relu'),
                                  keras.layers.Dense(32,activation='relu'),
                                  keras.layers.Dense(6,activation='softmax')])

    classifier.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    history = classifier.fit(train_data,
          steps_per_epoch=100,
          validation_data=test_data,
          validation_steps=50,
          epochs=epochs)
    
    acc = pd.DataFrame.from_dict(history.history)
    acc = pd.concat([pd.Series(range(0,epochs),name='epochs'),acc],axis=1)
    
    fig,(ax,ax1) = plt.subplots(nrows=2,ncols=1,figsize=(16,16))
    sns.lineplot(x='epochs',y='accuracy',data=acc,ax=ax,color='m')
    sns.lineplot(x='epochs',y='val_accuracy',data=acc,ax=ax,color='c')
    sns.lineplot(x='epochs',y='loss',data=acc,ax=ax1,color='m')
    sns.lineplot(x='epochs',y='val_loss',data=acc,ax=ax1,color='c')
    ax.legend(labels=['Test Accuracy','Training Accuracy'])
    ax1.legend(labels=['Test Loss','Training Loss'])
    plt.show()
    
    accuracies_.append((name,("Validation Accuracy",history.history['val_accuracy'][epochs-1]),("Training Accuracy",history.history['accuracy'][epochs-1])))

    return classifier

adam_classifier = train(Adam(lr=0.001),train_data,test_data,"Adam",50)
rms_classifier = train(RMSprop(lr=0.0001),train_data,test_data,"RMSprop",60)
sgd_classifier = train(SGD(lr=0.0001,momentum=0.9, nesterov=True),train_data,test_data,"SGD",60)
accuracies_
classes = {v:k for k,v in test_data.class_indices.items()}
true_labels = []
row_labels = []
for i in range(len(test_data)):
    for val in test_data[i][1]:
        row_labels.append(val.argmax(axis=0))
    true_labels.extend(row_labels)
predictions_ = []
row_predictions = []
for i in range(len(test_data)):
    predictions = adam_classifier.predict(test_data[i][0])
    for val in predictions:
        row_predictions.append(val.argmax(axis=0))
    predictions_.extend(row_predictions)
top10 = [next(test_data) for _ in range(40,50)]
adam_predictions = adam_classifier.predict(top10[0][0])
fig = plt.figure(figsize=(15,15))
for i,img in enumerate(top10):
    image = img[0][i]
    labels = classes[img[1][i].argmax(axis=0)]
    pred_labels = classes[adam_predictions[i].argmax(axis=0)]
    ax = fig.add_subplot(5,2,i+1)
    plt.subplots_adjust(hspace = .5)
    color = 'green' if labels == pred_labels else 'red'
    ax.set_title(f"Original : {labels} | Predicted : {pred_labels}",color=color)
    ax.imshow(image)
from sklearn.metrics import classification_report

print(classification_report(true_labels,predictions_,target_names=list(classes.values())))
rms_predictions = rms_classifier.predict(top10[0][0])
fig = plt.figure(figsize=(15,15))
for i,img in enumerate(top10):
    image = img[0][i]
    labels = classes[img[1][i].argmax(axis=0)]
    pred_labels = classes[rms_predictions[i].argmax(axis=0)]
    ax = fig.add_subplot(5,2,i+1)
    plt.subplots_adjust(hspace = .5)
    color = 'green' if labels == pred_labels else 'red'
    ax.set_title(f"Original : {labels} | Predicted : {pred_labels}",color=color)
    ax.imshow(image)
rms_predictions_ = []
row_predictions = []
for i in range(len(test_data)):
    predictions = rms_classifier.predict(test_data[i][0])
    for val in predictions:
        row_predictions.append(val.argmax(axis=0))
    rms_predictions_.extend(row_predictions)
print(classification_report(true_labels,rms_predictions_,target_names=list(classes.values())))
sgd_predictions = sgd_classifier.predict(top10[0][0])
fig = plt.figure(figsize=(15,15))
for i,img in enumerate(top10):
    image = img[0][i]
    labels = classes[img[1][i].argmax(axis=0)]
    pred_labels = classes[sgd_predictions[i].argmax(axis=0)]
    ax = fig.add_subplot(5,2,i+1)
    plt.subplots_adjust(hspace = .5)
    color = 'green' if labels == pred_labels else 'red'
    ax.set_title(f"Original : {labels} | Predicted : {pred_labels}",color=color)
    ax.imshow(image)
sgd_predictions_ = []
row_predictions = []
for i in range(len(test_data)):
    predictions = adam_classifier.predict(test_data[i][0])
    for val in predictions:
        row_predictions.append(val.argmax(axis=0))
    sgd_predictions_.extend(row_predictions)
print(classification_report(true_labels,sgd_predictions_,target_names=list(classes.values())))
import cv2
dir_path = "../input/intel-image-classification/seg_pred/seg_pred"
images = []
files = os.listdir(dir_path)
for fi in files:
    img = plt.imread(os.path.join(dir_path,fi))
    img = cv2.resize(img,(128,128))
    images.append(img)
plt.imshow(images[1])
images = np.array(images)
len(images)
images.shape
images_split = np.array_split(images,228)
images_split = np.array(images_split)
images_split[5].shape
pred_predictions = adam_classifier.predict(images_split[5])
predictions_classes = []
for preds in pred_predictions:
    predictions_classes.append(classes[preds.argmax(axis=0)])
images_split[5].shape
fig = plt.figure(figsize=(15,15))
for i,img in enumerate(images_split[5]):
    image = img
    labels = predictions_classes[i]
    ax = fig.add_subplot(8,4,i+1)
    plt.subplots_adjust(hspace = .5)
    ax.set_title(labels)
    ax.imshow(image)
