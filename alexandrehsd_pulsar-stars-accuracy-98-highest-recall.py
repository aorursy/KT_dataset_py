# import the necessary packages

from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



from keras.models import Sequential

from keras.layers.core import Dense, Dropout

from keras.optimizers import SGD

from keras import regularizers



import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
# Loading the dataset

dataset = pd.read_csv('../input/pulsar_stars.csv')
# EDA

print(dataset.head())
# Droping the target and assigning the rest to the data variable

data = dataset.drop(['target_class'], axis=1)



# Standardization procedure

scaler = StandardScaler()

data = scaler.fit_transform(data)



target = dataset[['target_class']]
# Construct the training and testing splits 

trainX, testX, trainY, testY = train_test_split(data, target, test_size=0.25)
lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.transform(testY)
# Defining the model

model = Sequential()

model.add(Dense(4, input_shape=(8,), activation='sigmoid'))

model.add(Dense(2, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))
sgd = SGD(0.12, momentum=0.4)



model.compile(loss='binary_crossentropy', optimizer=sgd,

    metrics=["accuracy"])



class_weight = {0 : 1., 1 : 2.}



H = model.fit(trainX, trainY, validation_data=(testX, testY), 

              batch_size=128, epochs=200, class_weight=class_weight, verbose=0)



scores = model.evaluate(testX, testY, verbose = 0)
predictions = model.predict(testX, batch_size=128)



# apply a step function to threshold the outputs to binary

# class labels

predictions[predictions < 0.5] = 0

predictions[predictions >= 0.5] = 1



report = classification_report(testY, predictions, 

                               target_names=['Non-pulsar Star', 'Pulsar Star'])



print('Accuracy = {:.7f}'.format(scores[1]))

print(report)
conf_matrix = confusion_matrix(testY, predictions)



# Plot the confusion matrix

import seaborn as sns

import matplotlib.pyplot as plt     



plt.figure(figsize=(10,8))

ax = plt.subplot()

sns.heatmap(conf_matrix, annot=True, ax = ax, fmt='d') #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])

ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])

#plt.savefig('confusion_matrix_wcw.png')
# Plotting the curve Epoch vs. Loss/Accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 200), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, 200), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 200), H.history["acc"], label="train_acc")

plt.plot(np.arange(0, 200), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(testY, predictions)



auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()