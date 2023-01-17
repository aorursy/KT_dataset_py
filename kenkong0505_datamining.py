from __future__ import print_function

%matplotlib inline



import os

import sys

import numpy as np

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm



from sklearn.model_selection import train_test_split

import keras

from keras import metrics

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical



# Add your code here

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.models import Model, Input

from keras.layers import Add, GlobalAveragePooling2D



from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, accuracy_score

from sklearn.utils.fixes import signature

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



img_width = 150

img_height = 150

num_classes = 2

DATA_DIR = '../input/data/data/'

image_filenames = [DATA_DIR+i for i in os.listdir(DATA_DIR)] # use this for full dataset





classes = [0 if 'cat' in filename else 1 for filename in image_filenames]

classes = to_categorical(np.asarray(classes))



images = np.array([cv2.resize(cv2.imread(filename), (img_width, img_height))

                   for filename in tqdm(image_filenames, total=len(image_filenames))],

                  dtype=np.float32)

images = preprocess_input(images)

    

X_train, X_validation, Y_train, Y_validation= train_test_split(images,classes,test_size=0.2,random_state=1)



X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation,Y_validation,test_size=0.5,random_state=1)



def BatchActivate(x):

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    return x



def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):

    x = Conv2D(filters, size, strides=strides, padding=padding)(x)

    if activation:

        x = BatchActivate(x)

    return x



def residual_block(blockInput, num_filters=16, batch_activate=False):

    x = BatchActivate(blockInput)

    x = convolution_block(x, num_filters, (3, 3))

    x = convolution_block(x, num_filters, (3, 3), activation=False)

    x = Add()([x, blockInput])

    if batch_activate:

        x = BatchActivate(x)

    return x



# the simplified resnet CNN model 

start_filters = 64

inputs = Input((img_height, img_width, 3))



x = Conv2D(start_filters, (3, 3), activation=None, padding='same')(inputs)

x = residual_block(x, start_filters * 1)

x = residual_block(x, start_filters * 1, True)

x = MaxPooling2D((2, 2))(x)



x = Conv2D(start_filters * 2, (3, 3), activation=None, padding='same')(x)

x = residual_block(x, start_filters * 2)

x = residual_block(x, start_filters * 2, True)

x = MaxPooling2D((2, 2))(x)



x = Conv2D(start_filters * 4, (3, 3), activation=None, padding='same')(x)

x = residual_block(x, start_filters * 4)

x = residual_block(x, start_filters * 4, True)

x = MaxPooling2D((2, 2))(x)



x = Conv2D(start_filters * 8, (3, 3), activation=None, padding='same')(x)

x = residual_block(x, start_filters * 8)

x = residual_block(x, start_filters * 8, True)

x = MaxPooling2D((2, 2))(x)



x = Conv2D(start_filters * 16, (3, 3), activation=None, padding='same')(x)

x = residual_block(x, start_filters * 16, True)



x = GlobalAveragePooling2D()(x)

outputs = Dense(num_classes, activation='softmax')(x)



model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# fit training data to the resnet CNN model

tuned_history = model.fit(X_train, Y_train,

                        batch_size=64,

                        epochs=60,

                        validation_data=(X_validation, Y_validation),

                        verbose=1,

                        callbacks=[

                            ModelCheckpoint("model.h5", monitor='val_acc', save_best_only=True, verbose=1),

                            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1),

                        ])



from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, accuracy_score

from sklearn.utils.fixes import signature



# Add your code here

tuned_pred = model.predict(X_test)

tuned_pred = np.around(tuned_pred)

tuned_pred = tuned_pred[:,1]

Y_test = Y_test[:,1]





# Referenced from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

# summarize history for accuracy

plt.plot(tuned_history.history['acc'])

plt.plot(tuned_history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(tuned_history.history['loss'])

plt.plot(tuned_history.history['val_loss'])

plt.title('Model Accuracy')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



print(classification_report(Y_test, tuned_pred))





#confusion matrix

print(confusion_matrix(Y_test,tuned_pred))



#precision-recall curve

prec, rec, thresholds = precision_recall_curve(Y_test, tuned_pred)

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(rec, prec, color='b', alpha=0.2,

         where='post')

plt.fill_between(rec, prec, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.0])

plt.xlim([0.0, 1.0])

plt.title("Precision-Recall curve")

plt.show()





#roc curve

fpr, tpr, thresholds = roc_curve(Y_test,tuned_pred)

plt.plot(fpr,tpr)

plt.title('ROC')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.show()



#auc value

print ("AUC value {}".format(auc(fpr,tpr)))



#accuracy_score

print ("Accuracy Score: {}".format(accuracy_score(Y_test,tuned_pred)))