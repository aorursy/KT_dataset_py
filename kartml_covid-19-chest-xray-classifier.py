# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
import cv2
import numpy as np
import argparse
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.config.list_physical_devices('GPU') 
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.layers import AveragePooling2D, Dropout,Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix
import seaborn as sn
import os
import itertools
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Add, Activation, Multiply, concatenate
from tensorflow.keras.applications import VGG16,DenseNet201,ResNet50,VGG19,Xception,InceptionResNetV2,InceptionV3
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,GlobalAveragePooling2D
train_path='../input/newdataset/final_dataset/train'
test_path='../input/newdataset/final_dataset/test'
val_path='../input/newdataset/final_dataset/val'
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(train_path,target_size = (320, 320),shuffle=True,seed=42,class_mode="categorical",color_mode = 'rgb',batch_size = 16)
test_generator = test_datagen.flow_from_directory(test_path,target_size = (320, 320),color_mode = 'rgb',batch_size = 1,seed=42,class_mode="categorical",shuffle = False)
val_generator = test_datagen.flow_from_directory(val_path,target_size = (320, 320),color_mode = 'rgb',batch_size = 1,seed=42,class_mode="categorical",shuffle = False)
def plot_img(img_arr):
  fig,axes=plt.subplots(2,8,figsize=(10,10))
  axes=axes.flatten()
  for img,ax in zip(img_arr,axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()
imgs,labels=next(train_generator)
plot_img(imgs)
print(labels)
#RANDOM INITIALIZATION OF WEIGHTS
class_weight = {
    0:0.85,
    1:0.15
}
lr = 1e-4
EPOCHS = 100
BS = 16
k_fold=2
cv_scores, model_history = list(), list()
for i in range(k_fold):
    
    print("K-FOLD:",i+1)
    baseModel = ResNet50(include_top=False,weights='imagenet',input_tensor=Input(shape=(320, 320, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model_1 = Model(inputs=baseModel.input, outputs=headModel)
    print("[INFO] compiling model...")
    opt = Adam(lr=lr,decay=lr/EPOCHS)
    model_1.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    print("[INFO] training head...")
    H1 = model_1.fit(train_generator,
                    steps_per_epoch = 320//BS,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = 26,
                   class_weight=class_weight)
    predictions=model_1.predict(test_generator,use_multiprocessing=True)
    predIdxs = np.argmax(predictions, axis=1)
    cm=confusion_matrix(y_true=test_generator.classes,y_pred=predIdxs)
    print(cm)
    print(classification_report(test_generator.classes,predIdxs))
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    cv_scores.append(acc)
    model_history.append(H1.history)

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
# from sklearn.metrics import plot_confusion_matrix
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(20,20))
    plt.show()
    
y_pred = np.argmax(predictions, axis=1)
cm=confusion_matrix(test_generator.classes, y_pred)
target_names = ['COVID19','OTHERS']
plot_confusion_matrix(cm,target_names)

from sklearn.metrics import roc_curve,auc
# fpr, tpr, thresholds = roc_curve(test_generator.classes, predIdxs)
# roc_auc = auc(fpr,tpr)

# plt.plot(fpr, tpr,label=f'ROC curve (area = {roc_auc})')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.rcParams['font.size'] = 12
# plt.title('ROC curve for COVID19 classifier')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.grid(True)

fpr = dict()
tpr = dict()

for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs)
    roc_auc = auc(fpr[i],tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2,label=f'ROC curve (area = {roc_auc:.3f})')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.title("ROC curve for COVID19 classifier")
plt.grid(True)
plt.show()
from sklearn.metrics import precision_recall_curve
precision = dict()
recall = dict()
for i in range(1):
    precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
    plt.plot(recall[i], precision[i], lw=2)

plt.xlabel("recall")
plt.ylabel("precision")
plt.title("prec")
print('Classification Report')
target_names = ['COVID19','OTHERS']
print(classification_report(test_generator.classes, predIdxs, target_names=target_names))
lr = 1e-4
EPOCHS = 100
BS = 16
k_fold=2
cv_scores, model_history = list(), list()
for i in range(k_fold):
    
    print("K-FOLD:",i+1)
    baseModel = VGG19(include_top=False,weights='imagenet',input_tensor=Input(shape=(320, 320, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)#pool_size=(4, 4)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model_2 = Model(inputs=baseModel.input, outputs=headModel)
    print("[INFO] compiling model...")
    opt = Adam(lr=lr,decay=lr/EPOCHS)
    model_2.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    print("[INFO] training head...")
    H2 = model_2.fit(train_generator,
                    steps_per_epoch = 320//BS,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = 26,
                   class_weight=class_weight)
    predictions_1=model_2.predict(test_generator,use_multiprocessing=True)
    predIdxs_1 = np.argmax(predictions_1, axis=1)
    cm=confusion_matrix(y_true=test_generator.classes,y_pred=predIdxs_1)
    print(cm)
    print(classification_report(test_generator.classes,predIdxs_1))
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    cv_scores.append(acc)
    model_history.append(H2.history)

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

# # fpr, tpr, thresholds = metrics.roc_curve(test_generator.classes, predIdxs)

# # plt.plot(fpr, tpr)
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.0])
# # plt.rcParams['font.size'] = 12
# # plt.title('ROC curve for COVID19 classifier')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.grid(True)

# # roc curve
# fpr = dict()
# tpr = dict()

# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_1)
#     plt.plot(fpr[i], tpr[i], lw=2)

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="best")
# plt.title("ROC curve for COVID19 classifier")
# plt.grid(True)
# plt.show()
# # from sklearn.metrics import plot_confusion_matrix
# def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
    
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="red" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.figure(figsize=(20,20))
#     plt.show()
    
# y_pred = np.argmax(predictions, axis=1)
# cm=confusion_matrix(test_generator.classes, y_pred)
# target_names = ['COVID19','OTHERS']
# plot_confusion_matrix(cm,target_names)
# from sklearn.metrics import precision_recall_curve
# precision = dict()
# recall = dict()
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
#     plt.plot(recall[i], precision[i], lw=2)

# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.title("prec")
# print('Classification Report')
# target_names = ['COVID19','OTHERS']
# print(classification_report(test_generator.classes, predIdxs, target_names=target_names))
lr = 1e-4
EPOCHS = 100
BS = 16
k_fold=1
cv_scores, model_history = list(), list()
for i in range(k_fold):
    
    print("K-FOLD:",i+1)
    baseModel = VGG16(include_top=False,weights='imagenet',input_tensor=Input(shape=(320, 320, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)#pool_size=(4, 4)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model_3 = Model(inputs=baseModel.input, outputs=headModel)
    print("[INFO] compiling model...")
    opt = Adam(lr=lr,decay=lr/EPOCHS)
    model_3.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    print("[INFO] training head...")
    H3 = model_3.fit(train_generator,
                    steps_per_epoch = 320//BS,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = 26,
                   class_weight=class_weight)
    predictions_2=model_3.predict(test_generator,use_multiprocessing=True)
    predIdxs_2 = np.argmax(predictions_2, axis=1)
    cm=confusion_matrix(y_true=test_generator.classes,y_pred=predIdxs_2)
    print(cm)
    print(classification_report(test_generator.classes,predIdxs_2))
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    cv_scores.append(acc)
    model_history.append(H3.history)

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
# # from sklearn.metrics import plot_confusion_matrix
# def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
    
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="red" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.figure(figsize=(20,20))
#     plt.show()
    
# y_pred = np.argmax(predictions, axis=1)
# cm=confusion_matrix(test_generator.classes, y_pred)
# target_names = ['COVID19','OTHERS']
# plot_confusion_matrix(cm,target_names)
# from sklearn.metrics import precision_recall_curve
# precision = dict()
# recall = dict()
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
#     plt.plot(recall[i], precision[i], lw=2)

# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.title("prec")
# # from sklearn import metrics
# # fpr, tpr, thresholds = metrics.roc_curve(test_generator.classes, predIdxs_2)

# # plt.plot(fpr, tpr)
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.0])
# # plt.rcParams['font.size'] = 12
# # plt.title('ROC curve for COVID19 classifier')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.grid(True)

# fpr = dict()
# tpr = dict()

# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_2)
#     roc_auc = auc(fpr[i],tpr[i])
#     plt.plot(fpr[i], tpr[i], lw=2,label=f'ROC curve (area = {roc_auc:.3f})')

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="best")
# plt.title("ROC curve for COVID19 classifier")
# plt.grid(True)
# plt.show()
# print('Classification Report')
# target_names = ['COVID19','OTHERS']
# print(classification_report(test_generator.classes, predIdxs, target_names=target_names))
lr = 1e-4
EPOCHS = 100
BS = 16
k_fold=1
cv_scores, model_history = list(), list()
for i in range(k_fold):
    
    print("K-FOLD:",i+1)
    baseModel = Xception(include_top=False,weights='imagenet',input_tensor=Input(shape=(320, 320, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)#pool_size=(4, 4)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model_4 = Model(inputs=baseModel.input, outputs=headModel)
    print("[INFO] compiling model...")
    opt = Adam(lr=lr,decay=lr/EPOCHS)
    model_4.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    print("[INFO] training head...")
    H4 = model_4.fit(train_generator,
                    steps_per_epoch = 320//BS,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = 26,
                   class_weight=class_weight)
    predictions_3=model_4.predict(test_generator,use_multiprocessing=True)
    predIdxs_3 = np.argmax(predictions_3, axis=1)
    cm=confusion_matrix(y_true=test_generator.classes,y_pred=predIdxs_3)
    print(cm)
    print(classification_report(test_generator.classes,predIdxs_3))
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    cv_scores.append(acc)
    model_history.append(H4.history)

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
# # from sklearn import metrics
# # fpr, tpr, thresholds = metrics.roc_curve(test_generator.classes, predIdxs)

# # plt.plot(fpr, tpr)
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.0])
# # plt.rcParams['font.size'] = 12
# # plt.title('ROC curve for COVID19 classifier')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.grid(True)

# fpr = dict()
# tpr = dict()

# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_3)
#     plt.plot(fpr[i], tpr[i], lw=2)

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="best")
# plt.title("ROC curve for COVID19 classifier")
# plt.grid(True)
# plt.show()
# # from sklearn.metrics import plot_confusion_matrix
# def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
    
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="red" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.figure(figsize=(20,20))
#     plt.show()
    
# y_pred = np.argmax(predictions, axis=1)
# cm=confusion_matrix(test_generator.classes, y_pred)
# target_names = ['COVID19','OTHERS']
# plot_confusion_matrix(cm,target_names)
# from sklearn.metrics import precision_recall_curve
# precision = dict()
# recall = dict()
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
#     plt.plot(recall[i], precision[i], lw=2)

# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.title("prec")
# print('Classification Report')
# target_names = ['COVID19','OTHERS']
# print(classification_report(test_generator.classes, predIdxs, target_names=target_names))
lr = 1e-4
EPOCHS = 100
BS = 16
k_fold=1
cv_scores, model_history = list(), list()
for i in range(k_fold):
    
    print("K-FOLD:",i+1)
    baseModel = DenseNet201(include_top=False,weights='imagenet',input_tensor=Input(shape=(320, 320, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)#pool_size=(4, 4)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model_5 = Model(inputs=baseModel.input, outputs=headModel)
    print("[INFO] compiling model...")
    opt = Adam(lr=lr,decay=lr/EPOCHS)
    model_5.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    print("[INFO] training head...")
    H5 = model_5.fit(train_generator,
                    steps_per_epoch = 320//BS,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = 26,
                   class_weight=class_weight)
    predictions_4=model_5.predict(test_generator,use_multiprocessing=True)
    predIdxs_4 = np.argmax(predictions_4, axis=1)
    cm=confusion_matrix(y_true=test_generator.classes,y_pred=predIdxs_4)
    print(cm)
    print(classification_report(test_generator.classes,predIdxs_4))
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    cv_scores.append(acc)
    model_history.append(H5.history)

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
# # from sklearn import metrics
# # fpr, tpr, thresholds = metrics.roc_curve(test_generator.classes, predIdxs_4)

# # plt.plot(fpr, tpr)
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.0])
# # plt.rcParams['font.size'] = 12
# # plt.title('ROC curve for COVID19 classifier')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.grid(True)

# fpr = dict()
# tpr = dict()

# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_4)
#     roc_auc = auc(fpr[i],tpr[i])
#     plt.plot(fpr[i], tpr[i], lw=2,label=f'ROC curve (area = {roc_auc:.3f})')

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="best")
# plt.title("ROC curve for COVID19 classifier")
# plt.grid(True)
# plt.show()
# # from sklearn.metrics import plot_confusion_matrix
# def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
    
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="red" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.figure(figsize=(20,20))
#     plt.show()
    
# y_pred = np.argmax(predictions, axis=1)
# cm=confusion_matrix(test_generator.classes, y_pred)
# target_names = ['COVID19','OTHERS']
# plot_confusion_matrix(cm,target_names)
# from sklearn.metrics import precision_recall_curve
# precision = dict()
# recall = dict()
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
#     plt.plot(recall[i], precision[i], lw=2)

# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.title("prec")
# print('Classification Report')
# target_names = ['COVID19','OTHERS']
# print(classification_report(test_generator.classes, predIdxs, target_names=target_names))
lr = 1e-4
EPOCHS = 100
BS = 16
k_fold=1
cv_scores, model_history = list(), list()
for i in range(k_fold):
    
    print("K-FOLD:",i+1)
    baseModel = InceptionResNetV2(include_top=False,weights='imagenet',input_tensor=Input(shape=(320, 320, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)#pool_size=(4, 4)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model_6 = Model(inputs=baseModel.input, outputs=headModel)
    print("[INFO] compiling model...")
    opt = Adam(lr=lr,decay=lr/EPOCHS)
    model_6.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    print("[INFO] training head...")
    H6 = model_6.fit(train_generator,
                    steps_per_epoch = 320//BS,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = 26,
                   class_weight=class_weight)
    predictions_5=model_6.predict(test_generator,use_multiprocessing=True)
    predIdxs_5 = np.argmax(predictions_5, axis=1)
    cm=confusion_matrix(y_true=test_generator.classes,y_pred=predIdxs_5)
    print(cm)
    print(classification_report(test_generator.classes,predIdxs_5))
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    cv_scores.append(acc)
    model_history.append(H6.history)

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
# # from sklearn import metrics
# # fpr, tpr, thresholds = metrics.roc_curve(test_generator.classes, predIdxs_5)

# # plt.plot(fpr, tpr)
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.0])
# # plt.rcParams['font.size'] = 12
# # plt.title('ROC curve for COVID19 classifier')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.grid(True)

# fpr = dict()
# tpr = dict()

# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_5)
#     roc_auc = auc(fpr[i],tpr[i])
#     plt.plot(fpr[i], tpr[i], lw=2,label=f'ROC curve (area = {roc_auc:.3f})')

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="best")
# plt.title("ROC curve for COVID19 classifier")
# plt.grid(True)
# plt.show()
# # from sklearn.metrics import plot_confusion_matrix
# def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
    
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="red" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.figure(figsize=(20,20))
#     plt.show()
    
# y_pred = np.argmax(predictions, axis=1)
# cm=confusion_matrix(test_generator.classes, y_pred)
# target_names = ['COVID19','OTHERS']
# plot_confusion_matrix(cm,target_names)
# from sklearn.metrics import precision_recall_curve
# precision = dict()
# recall = dict()
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
#     plt.plot(recall[i], precision[i], lw=2)

# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.title("prec")
# print('Classification Report')
# target_names = ['COVID19','OTHERS']
# print(classification_report(test_generator.classes, predIdxs, target_names=target_names))
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.figure(figsize=(10,10))
#loss
plt.plot(np.arange(0, N), H1.history["loss"], label="Resnet50_train_loss")
# plt.plot(np.arange(0, N), H1.history["accuracy"], label="Resnet50_train_acc")

plt.plot(np.arange(0, N), H2.history["loss"], label="VGG_19_train_loss")
# plt.plot(np.arange(0, N), H2.history["accuracy"], label="VGG_19_train_acc")

# plt.plot(np.arange(0, N), H3.history["loss"], label="VGG_16_train_loss")
# # plt.plot(np.arange(0, N), H3.history["accuracy"], label="VGG_16_train_acc")

# plt.plot(np.arange(0, N), H4.history["loss"], label="Xception_train_loss")
# # plt.plot(np.arange(0, N), H4.history["accuracy"], label="Xception_train_acc")

# plt.plot(np.arange(0, N), H5.history["loss"], label="Densenet_201_train_loss")
# # plt.plot(np.arange(0, N), H5.history["accuracy"], label="Densenet_201_train_acc")

# plt.plot(np.arange(0, N), H6.history["loss"], label="InceptionResNetV2_train_loss")
# # plt.plot(np.arange(0, N), H6.history["accuracy"], label="InceptionResNetV2_train_acc")

plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.show()
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.figure(figsize=(10,10))
#loss
# plt.plot(np.arange(0, N), H1.history["loss"], label="Resnet50_train_loss")
plt.plot(np.arange(0, N), H1.history["accuracy"], label="Resnet50_train_acc")

# plt.plot(np.arange(0, N), H2.history["loss"], label="VGG_19_train_loss")
plt.plot(np.arange(0, N), H2.history["accuracy"], label="VGG_19_train_acc")

# # plt.plot(np.arange(0, N), H3.history["loss"], label="VGG_16_train_loss")
# plt.plot(np.arange(0, N), H3.history["accuracy"], label="VGG_16_train_acc")

# # plt.plot(np.arange(0, N), H4.history["loss"], label="Xception_train_loss")
# plt.plot(np.arange(0, N), H4.history["accuracy"], label="Xception_train_acc")

# # plt.plot(np.arange(0, N), H5.history["loss"], label="Densenet_201_train_loss")
# plt.plot(np.arange(0, N), H5.history["accuracy"], label="Densenet_201_train_acc")

# # plt.plot(np.arange(0, N), H6.history["loss"], label="InceptionResNetV2_train_loss")
# plt.plot(np.arange(0, N), H6.history["accuracy"], label="InceptionResNetV2_train_acc")

plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.figure(figsize=(10,10))
plt.plot(np.arange(0, N), H1.history["val_loss"], label="Resnet50_val_loss")
# plt.plot(np.arange(0, N), H1.history["val_accuracy"], label="Resnet50_val_acc")

plt.plot(np.arange(0, N), H2.history["val_loss"], label="VGG_19_val_loss")
# plt.plot(np.arange(0, N), H2.history["val_accuracy"], label="VGG_19_val_acc")

# plt.plot(np.arange(0, N), H3.history["val_loss"], label="VGG_16_val_loss")
# # plt.plot(np.arange(0, N), H3.history["val_accuracy"], label="VGG_16_val_acc")

# plt.plot(np.arange(0, N), H4.history["val_loss"], label="Xception_val_loss")
# # plt.plot(np.arange(0, N), H4.history["val_accuracy"], label="Xception_val_acc")

# plt.plot(np.arange(0, N), H5.history["val_loss"], label="Densenet_201_val_loss")
# # plt.plot(np.arange(0, N), H5.history["val_accuracy"], label="Densenet_201_val_acc")


# plt.plot(np.arange(0, N), H6.history["val_loss"], label="InceptionResNetV2_val_loss")
# # plt.plot(np.arange(0, N), H6.history["val_accuracy"], label="InceptionResNetV2_val_acc")

plt.title("Validation Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Validation Loss")
plt.legend()
plt.show()
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.figure(figsize=(10,10))
# plt.plot(np.arange(0, N), H1.history["val_loss"], label="Resnet50_val_loss")
plt.plot(np.arange(0, N), H1.history["val_accuracy"], label="Resnet50_val_acc")

# plt.plot(np.arange(0, N), H2.history["val_loss"], label="VGG_19_val_loss")
plt.plot(np.arange(0, N), H2.history["val_accuracy"], label="VGG_19_val_acc")

# # plt.plot(np.arange(0, N), H3.history["val_loss"], label="VGG_16_val_loss")
# plt.plot(np.arange(0, N), H3.history["val_accuracy"], label="VGG_16_val_acc")

# # plt.plot(np.arange(0, N), H4.history["val_loss"], label="Xception_val_loss")
# plt.plot(np.arange(0, N), H4.history["val_accuracy"], label="Xception_val_acc")

# # plt.plot(np.arange(0, N), H5.history["val_loss"], label="Densenet_201_val_loss")
# plt.plot(np.arange(0, N), H5.history["val_accuracy"], label="Densenet_201_val_acc")


# # plt.plot(np.arange(0, N), H6.history["val_loss"], label="InceptionResNetV2_val_loss")
# plt.plot(np.arange(0, N), H6.history["val_accuracy"], label="InceptionResNetV2_val_acc")

plt.title("Validation Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
from sklearn.metrics import roc_curve,auc
fpr = dict()
tpr = dict()
plt.figure(figsize=(10,10))

for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs)
    roc_auc = auc(fpr[i],tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2,label=f'Resnet50:ROC curve (area = {roc_auc:.3f})',color='r')
for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_1)
    roc_auc = auc(fpr[i],tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2,label=f'VGG19:ROC curve (area = {roc_auc:.3f})',color='b')
# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_2)
#     roc_auc = auc(fpr[i],tpr[i])
#     plt.plot(fpr[i], tpr[i], lw=2,label=f'VGG16:ROC curve (area = {roc_auc:.3f})',color='g')
# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_3)
#     roc_auc = auc(fpr[i],tpr[i])
#     plt.plot(fpr[i], tpr[i], lw=2,label=f'Xception:ROC curve (area = {roc_auc:.3f})',color='k')
# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_4)
#     roc_auc = auc(fpr[i],tpr[i])
#     plt.plot(fpr[i], tpr[i], lw=2,label=f'DenseNet201:ROC curve (area = {roc_auc:.3f})',color='c')
# for i in range(1):
#     fpr[i], tpr[i], _ = roc_curve(test_generator.classes, predIdxs_5)
#     roc_auc = auc(fpr[i],tpr[i])
#     plt.plot(fpr[i], tpr[i], lw=2,label=f'InceptionResnetV2:ROC curve (area = {roc_auc:.3f})',color='m')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.title("ROC curve for COVID19 classifier")
plt.grid(True)
# plt.figure(figsize=(10,10))
plt.show()
from sklearn.metrics import precision_recall_curve
plt.figure(figsize=(10,10))
precision = dict()
recall = dict()
for i in range(1):
    precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
    plt.plot(recall[i], precision[i], lw=2,label="Resnet50",color='r')
for i in range(1):
    precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs_1)
    plt.plot(recall[i], precision[i], lw=2,label="VGG19",color='b')
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs_2)
#     plt.plot(recall[i], precision[i], lw=2,label="VGG16",color='g')
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs_3)
#     plt.plot(recall[i], precision[i], lw=2,label="Xception",color='k')
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs_4)
#     plt.plot(recall[i], precision[i], lw=2,label="DenseNet201",color='c')
    
# for i in range(1):
#     precision[i], recall[i],_= precision_recall_curve(test_generator.classes,predIdxs)
#     plt.plot(recall[i], precision[i], lw=2,label="InceptionResnetV2",color='m')
    
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("Precision/Recall Curve")
plt.grid(True)
plt.show()
