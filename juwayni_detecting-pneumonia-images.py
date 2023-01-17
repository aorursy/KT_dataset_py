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
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_dir ='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'  #Normal 1,341, Pneumonia 3,875, Total: 5,216 || if batch size = 32, then steps_per_epoch = 5,216/32 = 163    10000/32 = 313

validation_dir ='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/' #Normal 8, Pneumonia 8, Total: 16 || if batch size = 16, then steps_per_epoch = 16/16 = 1

test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/' #Normal 234, Pneumonia 390, Total: 624 if batch size = 16, then steps_per_epoch = 624/16 



train_datagenerator = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

        )

test_datagenerator = ImageDataGenerator(rescale=1. / 255)





train_generator = train_datagenerator.flow_from_directory(train_dir, target_size = (64,64), class_mode = 'binary', batch_size = 32)

validation_generator = train_datagenerator.flow_from_directory(validation_dir, target_size = (64,64), class_mode = 'binary', batch_size = 16)

test_generator = test_datagenerator.flow_from_directory(test_dir, target_size = (64,64), class_mode = 'binary', batch_size = 16, shuffle = False)

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dropout, Flatten, Dense

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.applications.vgg16 import VGG16



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))



top_model = Sequential()

top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  

top_model.add(Dense(128, activation='relu'))

top_model.add(Dense(1, activation='sigmoid'))



classifier = Model(inputs=base_model.input, outputs=top_model(base_model.output))

classifier.summary()

#freezing the first two  blocks of convolutional layer, based on the fact that layers closer to the output are more pre-data specific so we want to update the later layers and leave the early layers

classifier.get_layer('block1_conv1').trainable = False

classifier.get_layer('block1_conv2').trainable = False



classifier.get_layer('block2_conv1').trainable = False

classifier.get_layer('block2_conv2').trainable = False



classifier.summary()
# focal loss 

def focal_loss(alpha=0.25,gamma=2.0):

    def focal_crossentropy(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)



        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))



        alpha_factor = 1

        modulating_factor = 1



        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))

        modulating_factor = K.pow((1-p_t), gamma)



        # compute the final loss and return

        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)

    return focal_crossentropy



optimizer = RMSprop(0.0001)

classifier.compile(loss=focal_loss(), optimizer = optimizer,

                metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(patience=10, monitor = 'val_loss', verbose = 1)



classifier.fit_generator(train_generator, steps_per_epoch = 163, epochs = 50, verbose = 1, validation_data = validation_generator, validation_steps = 16 // 16, callbacks = [es])

score = classifier.evaluate_generator(test_generator, steps = 624 // 16)

test_generator.reset()

scores = classifier.predict_generator(test_generator, steps = 39)



correct = 0

for i, n in enumerate(test_generator.filenames):    #i is indexing starting from 0 || n is string type which takes the values in each row in the test_generator.filenames

    if n.startswith("NORMAL") and np.round(scores[i][0]) == 0:

        correct += 1

    if n.startswith("PNEUMONIA") and np.round(scores[i][0]) == 1:

        correct += 1

        

print("Correct:", correct, " Total: ", len(test_generator.filenames))

print('Test Accuracy = ', score[1] * 100, 'Test Loss = ', score[0] * 100)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

test_labels = test_generator.classes



cm = confusion_matrix(test_labels, np.round(scores[:,0]))

print(cm)



import matplotlib.pyplot as plt

from mlxtend.plotting import plot_confusion_matrix

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.xticks(range(1), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(1), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()



precision = np.round(precision_score(test_labels, np.round(scores[:,0])), 2) * 100

recall = np.round(recall_score(test_labels, np.round(scores[:,0])), 2) * 100    #This is an important metric because if we fail to diagnose people with pneumonia, that is have many False Negatives, the patient's conditions will get worse and they'll not be treated

f1_score = np.round(f1_score(test_labels, np.round(scores[:,0])), 2) * 100



print('Precision: ', precision, 'Recall: ', recall, 'F1 score: ', f1_score)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot



auc = roc_auc_score(test_labels, scores[:,0]) * 100

print('AUC:', auc, '%')

# calculate roc curve

fpr, tpr, thresholds = roc_curve(test_labels, scores[:,0])

# plot no skill

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr, marker='.')

# show the plot

pyplot.show()