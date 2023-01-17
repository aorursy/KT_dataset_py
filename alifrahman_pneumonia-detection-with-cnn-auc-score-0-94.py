import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout,Dense, Flatten, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Lambda, Input, ZeroPadding2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import itertools
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from itertools import cycle
from scipy import interp
from sklearn import metrics
#from imutils import paths
from keras.models import load_model
import os
import cv2
import shutil
from keras import backend as K
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline
img_width, img_height = 64, 64
train_data_dir = '../input/chest-xray-pneumonia/chest_xray/train'
#validation_data_dir = '../input/binary-wbc/binary_data/val'
test_data_dir = '../input/chest-xray-pneumonia/chest_xray/test'
nb_train_samples = 5216
nb_validation_samples = 624
#epochs = 10
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
train_datagen = ImageDataGenerator(
    rescale=1. / 255, #rescales each images (normalization)
    shear_range=0.2,  #shears each images by 0.2%
    zoom_range=0.2, # zoom each image by 0.2%
    width_shift_range=0.2,  #shifts width
    height_shift_range=0.2,  #shifts height
    horizontal_flip=False,  #horizontally flips the images
    vertical_flip=False)
test_datagen = ImageDataGenerator(rescale=1. / 255)  #no augmentation is performed for test set except for the normalization.

train_batches = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    #classes=['MONONUCLEAR', 'POLYNUCLEAR'])
    class_mode='categorical')  #class mode is set to binary as we are performing binary classification.
#valid_batches = test_datagen.flow_from_directory(
    #validation_data_dir,
    #target_size=(img_width, img_height),
    #batch_size=batch_size,
    #classes=['MONONUCLEAR', 'POLYNUCLEAR'])
    #color_mode="grayscale")
test_batches = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=624,
    #classes=['MONONUCLEAR', 'POLYNUCLEAR'])
    class_mode='categorical')   #class mode is set to binary as we are performing binary classification.
epochs = 25
INIT_LR = 0.0001
opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
# BUILD CONVOLUTIONAL NEURAL NETWORKS
nets = 5
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()
    model[j].add(BatchNormalization(input_shape=input_shape))
    model[j].add(Conv2D(32, (3, 3), padding='valid'))
    #model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    #model[j].add(Dropout(0.4))
    model[j].add(MaxPooling2D(pool_size=(2, 2)))
    model[j].add(Dropout(0.5))
    
    model[j].add(Conv2D(32, (5, 5), padding='same'))
    #model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(MaxPooling2D(pool_size=(2, 2)))
    
    model[j].add(Conv2D(64, (5, 5), padding='same'))
    #model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(MaxPooling2D(pool_size=(2, 2)))
    
    model[j].add(Conv2D(128, (5, 5), padding='same'))
    #model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(MaxPooling2D(pool_size=(2, 2)))
    
    model[j].add(Flatten())
    #model[j].add(Dense(128))
    #model[j].add(Activation('relu'))
    model[j].add(Dense(256))
    model[j].add(Activation('relu'))
    model[j].add(Dropout(0.5))
    #model[j].add(Dense(32))
    #model[j].add(Activation('relu'))

    model[j].add(Dense(2))
    model[j].add(Activation('softmax'))
    model[j].summary()

    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model[j].compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
# DECREASE LEARNING RATE EACH EPOCH
#annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets
for j in range(nets):
    #X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(train_batches,
        epochs = epochs, steps_per_epoch = nb_train_samples // batch_size,
        validation_data = test_batches,
        validation_steps = nb_validation_samples // batch_size,
        callbacks=[#annealer,
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4),
        tf.keras.callbacks.ModelCheckpoint(filepath = '/kaggle/working/model_{val_accuracy:.3f}.h5', save_best_only=True,
                                          save_weights_only=False, monitor='val_accuracy')
        ])  #, callbacks=[annealer]
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
test_imgs, test_labels = next(test_batches)
# ENSEMBLE PREDICTIONS AND SUBMIT
results = np.zeros( (624,2) ) 
for j in range(nets):
    results = results + model[j].predict_generator(test_imgs, steps=1, verbose=0)
results = np.argmax(results,axis = -1)
#results = pd.Series(results,name="Label")
#submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
#submission.to_csv("MNIST-CNN-ENSEMBLE.csv",index=False)
results
test_labels = np.argmax(test_labels,axis = -1)
test_labels
#test_model = load_model('../input/acc-9330/model_0.933.h5')
cm = confusion_matrix(y_true=test_labels, y_pred=results)
acc = accuracy_score(test_labels, results)*100
tn, fp, fn, tp = cm.ravel()
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {0:0.2f}%'.format(acc))
print('Precision: {0:0.2f}%'.format(precision))
print('Recall: {0:0.2f}%'.format(recall))
print('F1-score: {0:0.2f}'.format(2*precision*recall/(precision+recall)))
#print('Train acc: {0:0.2f}'.format(np.round((h.history['accuracy'][-1])*100, 2)))
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
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_plot_labels = ['NORMAL', 'PNEUMONIA']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='')
#confusion matrix-->correct identification = 202+380= 582
#                -->wrong identification = 32+10 = 42
#                --> accuracy = 582/(582 + 42) = 0.9326 or (93.26% accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_true=test_labels, y_pred=results))
# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(results,test_labels)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), results.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 2

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'cornflowerblue'])
for i, color in zip(range(2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
for j in range(nets):
    accs = history[j].history['accuracy']
    val_accs = history[j].history['val_accuracy']

    plt.title("For CNN: "+ str(j+1))
    plt.plot(range(len(accs)),accs, label = 'Training_accuracy')
    plt.plot(range(len(accs)),val_accs, label = 'Validation_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
for j in range(nets):
    loss = history[j].history['loss']
    val_loss = history[j].history['val_loss']

    plt.title("For CNN: "+ str(j+1))
    plt.plot(range(len(loss)),loss, label = 'Training_loss')
    plt.plot(range(len(loss)),val_loss, label = 'Validation_loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
