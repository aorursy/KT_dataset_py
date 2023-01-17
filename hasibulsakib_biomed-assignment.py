import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
#tf.keras.applications.mobilenet.preprocess_input

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense,BatchNormalization,SpatialDropout2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation

import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import classification_report, confusion_matrix
train_path = '../input/largectdata/train'
valid_path = '../input/largectdata/valid'
test_path = '../input/largectdata/test'
train_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=32)
valid_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=32)
test_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=32, shuffle=False)
label_names=list(test_batches.class_indices.keys())
print(label_names)
labelValues=list(test_batches.class_indices.values())
print(labelValues)
from keras.models import Sequential
vgg16_model_1 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
#vgg16_model_1.summary()

model=Sequential()

# All layers except the last 4
for layer in vgg16_model_1.layers[:-4]:
  model.add(layer)
#model.summary()

model.add(Flatten())
model.add(Dense(4096, activation='relu',name='fc1'))
model.add(Dropout(0.3))
model.add(Dense(4096, activation='relu',name='fc2'))
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu',name='fc3'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax',name='output'))

for layer in model.layers[:-12]:
  layer.trainable = False

model.summary()
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
from datetime import datetime
start = datetime.now()

history=model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=20,
          verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)
# evaluating the model
train_loss, train_acc = model.evaluate(train_batches, steps=16)
validation_loss, valid_acc = model.evaluate(valid_batches, steps=16)
test_loss, test_acc = model.evaluate(test_batches, steps=16)
print('Train: %.3f, Valid: %.3f, Test: %.3f' % (train_acc,valid_acc, test_acc))
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
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
 
plot_acc_loss(history)
#Confution Matrix and Classification Report
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=1))
print(cm)
import seaborn as sns
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm=cm, classes=label_names, title='Confusion Matrix')
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
# print(TN,TP,FN,FP)

#Sensitivity tells us what proportion of the positive class got correctly classified.
TPR=sensitivity=TP/(TP+FN)

#What proportion of positive predictions was actually correct
Precision=TP/(TP+FP)

#Specificity tells us what proportion of the negative class got correctly classified.
TNR=specificity=TN/(TN+FP)


print("sensitivity",sensitivity)
print("specificity",specificity)
print("Precision",Precision)
from sklearn.metrics import roc_curve
#y_pred_keras = keras_model.predict(test_batches).ravel()
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
fpr, tpr, thresholds = roc_curve(test_batches.classes,np.argmax(predictions, axis=-1))

from sklearn.metrics import auc
auc= auc(fpr, tpr)

print("fpr=",fpr)
print("tpr=",tpr)
print("thresholds=",thresholds)
print("auc=",auc)
plt.figure()
#plt.xlim(0, 0.2)
#plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC= {:.3f}'.format(auc))
plt.xlabel('False positive rate (TPR)')
plt.ylabel('True positive  (FPR)')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()