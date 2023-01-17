import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout,Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools
from sklearn import metrics
from keras.models import load_model
import os
import shutil
from keras import backend as K
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline
img_width, img_height = 120, 160

train_data_dir = '../input/train-test/train_test_binary_data/train'
#validation_data_dir = '../input/binary-wbc/binary_data/validation'
test_data_dir = '../input/train-test/train_test_binary_data/test'
nb_train_samples = 9920
nb_validation_samples = 2483
epochs = 30
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    #vertical_flip=True
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_batches = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=['MONONUCLEAR', 'POLYNUCLEAR'])
    #color_mode="grayscale")
#valid_batches = test_datagen.flow_from_directory(
    #validation_data_dir,
    #target_size=(img_width, img_height),
    #batch_size=batch_size,
    #classes=['MONONUCLEAR', 'POLYNUCLEAR'])
    #color_mode="grayscale")
test_batches = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=2483,
    classes=['MONONUCLEAR', 'POLYNUCLEAR'])
    #color_mode="grayscale")
#train_batches = ImageDataGenerator().flow_from_directory(directory=train_data_dir, target_size=(150,150), classes=['MONONUCLEAR', 'POLYNUCLEAR'], batch_size=batch_size)
#valid_batches = ImageDataGenerator().flow_from_directory(directory=validation_data_dir, target_size=(150,150), classes=['MONONUCLEAR', 'POLYNUCLEAR'], batch_size=batch_size)
#test_batches = ImageDataGenerator().flow_from_directory(directory=test_data_dir, target_size=(150,150), classes=['MONONUCLEAR', 'POLYNUCLEAR'], batch_size=400, shuffle=False)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
#model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dense(64))
#model.add(Activation('relu'))
model.add(Dropout(0.4))
#model.add(Dense(32))
#model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
%%time
h = model.fit_generator(
    train_batches,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_batches,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[
       # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4),
        tf.keras.callbacks.ModelCheckpoint(filepath = '/kaggle/working/model_{val_accuracy:.3f}.h5', save_best_only=True,
                                          save_weights_only=False, monitor='val_accuracy')
    ])
test_imgs, test_labels = next(test_batches)
#plots(test_imgs, rows=50, titles=test_labels)
#test_labels = test_labels[:,0]
rounded_labels = np.argmax(test_labels, axis=-1)
#test_labels
test_model = load_model('./model_0.978.h5')
predictions = test_model.predict_generator(test_batches, steps=1, verbose=0)
predictions
rounded_prediction = np.argmax(predictions, axis=-1)
for i in rounded_prediction:
    print(i)
cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_prediction)
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
cm_plot_labels = ['MONONUCLEAR', 'POLYNUCLEAR']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='confusion_matrix')
#confusion matrix-->correct identification = 1208+1220= 2428
#                -->wrong identification = 28+27 = 55
#                --> accuracy = 2428/(2428 + 55) = 0.9778 or (97.78% accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_true=rounded_labels, y_pred=rounded_prediction))
def plot_roc(rounded_prediction,rounded_labels):
    fpr, tpr, _ = roc_curve(rounded_prediction,rounded_labels)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristics (ROC)')
    plt.legend(loc='lower right')
    plt.show()
plot_roc(rounded_prediction,rounded_labels)
score = metrics.log_loss(rounded_labels,rounded_prediction)
print("Log loss score: {}".format(score))
accs = h.history['accuracy']
val_accs = h.history['val_accuracy']

plt.plot(range(len(accs)),accs, label = 'Training_accuracy')
plt.plot(range(len(accs)),val_accs, label = 'Validation_accuracy')
plt.legend()
plt.show()
accs = h.history['loss']
val_accs = h.history['val_loss']

plt.plot(range(len(accs)),accs, label = 'Training_loss')
plt.plot(range(len(accs)),val_accs, label = 'Validation_loss')
plt.legend()
plt.show()
