import tensorflow as tf

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
# Import Libraries

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

from sklearn.model_selection import train_test_split

import pandas as pd

import itertools

import pickle

# Keras API

from tensorflow import keras

from tensorflow.keras import Sequential, layers, optimizers, callbacks, models

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model
# Initializations

width=64

height=64

depth=3



_LR =0.0002

_Dropout = 0.1

BS = 32

_Epoch =46
# Import dataset pickle file

with open('../input/alexnet-dataset64x64/alexnet_dataset(64x64).pkl', 'rb') as f:

    augment_image_list, augment_label_list, test_image_list, test_label_list = pickle.load(f)
# Variables to store size of list of training, validation, and testing images

augment_image_size = len(augment_image_list)

test_image_size = len(test_image_list)

print(f"Image size from augmentation: {augment_image_size}")

print(f"Image size for testing: {test_image_size}")
# Variable to store binarized list of training, validation, and testing image labels

label_binarizer = LabelBinarizer()

augment_image_labels = label_binarizer.fit_transform(augment_label_list)

test_image_labels = label_binarizer.fit_transform(test_label_list)

# Variable to store number of classes

n_classes = len(label_binarizer.classes_)

print(f"Number of classes: {n_classes}")

# Display the classes

print(label_binarizer.classes_)
# Normalize values in train_image_list and test_image_list

np_augment_image_list = np.array(augment_image_list, dtype=np.float32) / 225.0

np_test_image_list = np.array(test_image_list, dtype=np.float32) / 225.0
#Alexnet

input_shape = (height, width, depth)

model = Sequential()

model.add(Conv2D(filters=6, kernel_size=(11,11), strides=2, input_shape=input_shape, activation="relu"))

model.add(MaxPooling2D(strides=1, padding="same"))

model.add(Conv2D(filters=8, kernel_size=(5,5), strides=1, activation="relu"))

model.add(MaxPooling2D(strides=1, padding="same"))

model.add(Conv2D(filters=12, kernel_size=(3,3), strides=1, activation="relu"))

model.add(Conv2D(filters=12, kernel_size=(3,3), strides=1, activation="relu"))

model.add(Conv2D(filters=8, kernel_size=(3,3), strides=1, activation="relu"))

model.add(MaxPooling2D(strides=2, padding="same"))

model.add(Flatten())

model.add(Dense(128,activation="relu"))

#model.add(Dropout(_Dropout))

model.add(Dense(128,activation="relu"))

#model.add(Dropout(_Dropout))

model.add(Dense(100,activation="relu"))

#model.add(Dropout(_Dropout))

model.add(Dense(n_classes,activation="softmax"))

model.summary()
# Compile the model

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = _LR), metrics=['accuracy'])
# Instantiating callbacks

mc = ModelCheckpoint('./alexnet_optimal.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
print("[INFO] Spliting data to train, validation")

x_train, x_valid, y_train, y_valid = train_test_split(np_augment_image_list, augment_image_labels, test_size=0.2, random_state = 42)
# Train the model

with tf.device('/device:GPU:0'):

  history = model.fit(x_train, y_train, batch_size=BS, validation_data=(x_valid, y_valid), 

                      steps_per_epoch=len(x_train) // BS, epochs=_Epoch, verbose=2, callbacks=[mc])
# Plot in a graph fashion

train_acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

train_loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

#Train and validation accuracy

plt.plot(epochs, train_acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.xlabel('Num of Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.figure()

#Train and validation loss

plt.plot(epochs, train_loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Num of Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Load the best model

saved_model = load_model('./alexnet_optimal.h5')
# Evaluate the model's accuracy

print("[INFO] Calculating training accuracy")

scores = saved_model.evaluate(x_valid, y_valid)

print(f"Average Train Loss: {scores[0]*100}")

print(f"Average Train Accuracy: {scores[1]*100}")
# Evaluate the model's accuracy against test set

print("[INFO] Calculating test accuracy")

scores = saved_model.evaluate(np_test_image_list, test_image_labels)

print(f"Average Test Loss: {scores[0]*100}")

print(f"Average Test Accuracy: {scores[1]*100}")
# sklearn Accuracy score

preds = saved_model.predict(np_test_image_list)

preds = preds>0.5 #convert preds to one digit (boolean)

acc_score = round((accuracy_score(test_image_labels, preds) * 100), 2)

print(f"accuracy score: {acc_score}")
# sklearn Precision score

prec = round((precision_score(test_image_labels, preds, average='micro') * 100), 2)

print(f"precision score: {prec}")
# sklearn Recall score

rec = round((recall_score(test_image_labels, preds, average='micro') * 100), 2)

print(f"recall score: {rec}")
# sklearn F1 score

f1 = round((f1_score(test_image_labels, preds, average='micro') * 100), 2)

print(f"f1 score: {f1}")
# Classification report on test set

preds_class_report = np.round(saved_model.predict(np_test_image_list),0)

classes = ["black sigatoka","bunchy top","healthy","panama wilt","yellow sigatoka"]

classification_metrics = classification_report(test_image_labels, preds_class_report, target_names=classes )

print(classification_metrics)
# Variables for confusion matrix

categorical_test_labels = pd.DataFrame(test_image_labels).idxmax(axis=1)

categorical_preds = pd.DataFrame(preds_class_report).idxmax(axis=1)
# Define the confusion matrix

confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)
# Function to plot the confusion matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    #Add Normalization Option

    '''prints pretty confusion metric with normalization option '''

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    

    #print(cm)

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Show confusion matrix without normalization

plot_confusion_matrix(confusion_matrix, classes)
# Show confusion matrix with normalization (percentage)

plot_confusion_matrix(confusion_matrix, classes, normalize=True)