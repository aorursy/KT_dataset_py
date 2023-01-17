import numpy as np

from random import randint

from sklearn.utils import shuffle

from sklearn.preprocessing import MinMaxScaler
#Generate own dataset given a imagined scenario:

    #a medicine is being tested on people ranging age from 13-100

    #50% people are of 13<age<65 and the rest 50% age<=65

    #5% of younger people experience side effects and 95% don't

    #95% of older people experience side effects and 5% don't



train_samples = []

train_labels = []
for i in range(50):

    random_younger = randint(13,64)

    train_samples.append(random_younger)

    train_labels.append(1)

    

    random_older = randint(65,100)

    train_samples.append(random_older)

    train_labels.append(0)

    

for i in range(1000):

    random_younger = randint(13,64)

    train_samples.append(random_younger)

    train_labels.append(0)

    

    random_older = randint(65,100)

    train_samples.append(random_older)

    train_labels.append(1)
for i in train_samples:

    print(i)
for i in train_labels:

    print(i)
train_samples = np.array(train_samples)

train_labels = np.array(train_labels)

train_labels, train_samples = shuffle(train_labels, train_samples)
#Normalizing

scaler = MinMaxScaler(feature_range=(0,1))

scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
for i in scaled_train_samples:

    print(i)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy
model = Sequential([

    Dense(units=16, input_shape=(1,), activation='relu'),

    Dense(units=32, activation='relu'),

    Dense(units=2, activation='softmax')

])
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)
#creating test set following the same process

test_samples = []

test_labels = []
for i in range(10):

    random_younger = randint(13,64)

    test_samples.append(random_younger)

    test_labels.append(1)

    

    random_older = randint(65,100)

    test_samples.append(random_older)

    test_labels.append(0)

    

for i in range(200):

    random_younger = randint(13,64)

    test_samples.append(random_younger)

    test_labels.append(0)

    

    random_older = randint(65,100)

    test_samples.append(random_older)

    test_labels.append(1)
test_labels = np.array(test_labels)

test_samples = np.array(test_samples)

test_labels, test_samples = shuffle(test_labels, test_samples)
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
for i in predictions:

    print (i)
rounded_prediction = np.argmax(predictions, axis=-1)
for i in rounded_prediction:

    print(i)
%matplotlib inline

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_prediction)
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
cm_plot_labels = ['no_side_effects','had_side_effects']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='confusion_matrix')
import os.path

if os.path.isfile('/kaggle/working/medical_trial_model.h5') is False:

    model.save('/kaggle/working/medical_trial_model.h5')

    

    

#saves the whole model including architecture, weights.Saves the whole state of the model
from tensorflow.keras.models import load_model

new_model = load_model('./medical_trial_model.h5')
new_model.summary()
new_model.get_weights()
new_model.optimizer
#saves only the architecture

json_string = model.to_json()
json_string
from tensorflow.keras.models import model_from_json

model_architecture = model_from_json(json_string)
model_architecture.summary()
#saves only the weights

import os.path

if os.path.isfile('/kaggle/working/medical_trial_model_weights.h5') is False:

    model.save_weights('/kaggle/working/medical_trial_model_weights.h5')
model2 = Sequential([

    Dense(units=16, input_shape=(1,), activation='relu'),

    Dense(units=32, activation='relu'),

    Dense(units=2, activation='softmax')

])
model2.load_weights('./medical_trial_model.h5')
model2.get_weights()