import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

import keras.backend as K



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import ast

from collections import Counter

import json

import os
annotations = []



baseLabelPath = '../input/labels/'

labelFiles = sorted(os.listdir(baseLabelPath))



with open(os.path.join(baseLabelPath, labelFiles[0])) as file:

    dictionary = ast.literal_eval(list(file)[0]) # json data

    for items in sorted(dictionary.items()):

        annotations.append((items[0], items[1]))



data = pd.read_csv('../input/trainingdata/TrainingData.csv')



data = data.iloc[: 593360, :]
data.head()
data.describe()
print(f'Data size : {data.shape}')

labels = []



cntLabels = Counter()



for index, anno in enumerate(annotations):

    cntLabels[anno[1]] += 1

    li = [0.0] * 10

    li[anno[1]] = 1

    labels.append(li)



labels = np.array(labels)

#labels = np.reshape(labels, (labels.shape[0] * 2, 10))

total = sum(cntLabels.values())





print(cntLabels.most_common())

print(f'Total labels: {total}')

print(f'Training Labels size: {labels.shape}')
classWeights = []



for index, num in enumerate(cntLabels):

    prob = cntLabels[index] / total

    classWeights.append((index, 1 / prob))



classWeights = dict(classWeights)
#data['label'] = labels

#data['label'].value_counts()
X_data = data.iloc[:, :]



X_data = np.array(X_data)

X_data = X_data.reshape((int(X_data.shape[0] / 16), 16 * 28))



y_data = labels



print(f'Total training samples: {X_data.shape[0]}')

print(f'Total features: {X_data.shape[1]}')

print(f'Total number of labels: {y_data.shape[1]}')
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.1, shuffle = True)
def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall

    



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
METRICS = [ 

      keras.metrics.BinaryAccuracy(name='accuracy'),

      keras.metrics.Precision(name='precision'),

      keras.metrics.Recall(name='recall')

]
def getModel():

    model = Sequential()



    model.add(Dense(512, input_shape=(448, ), name='dense1'))

    model.add(Dense(512, name='dense2'))

    model.add(Dense(512, name='dense3'))



    model.add(Dropout(0.3, name='dropout1'))



    model.add(Dense(256, name='dense4'))



    model.add(Dropout(0.2, name='dropout2'))



    model.add(Dense(128, name='dense5'))



    model.add(Dense(10, activation='softmax', name='output'))

    

    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])

    

    return model
model = getModel()

model.summary()
def stepDecay(epoch):

    if epoch <= 20:

        K.set_value(model.optimizer.lr, 1e-3)

    else:

        K.set_value(model.optimizer.lr, 1e-4)

        

    return K.get_value(model.optimizer.lr)
class SnapshotCallbackBuilder:

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.001):

        self.T = nb_epochs

        self.M = nb_snapshots

        self.alpha_zero = init_lr



    def get_callbacks(self, model_prefix='Model'):



        callback_list = [

            ModelCheckpoint("JointFC_model.h5",monitor='val_precision_m', 

                                   mode = 'max', save_best_only=True, verbose=1),

            swa,

            LearningRateScheduler(stepDecay)

        ]



        return callback_list
class SWA(keras.callbacks.Callback):

    

    def __init__(self, filepath, swa_epoch):

        super(SWA, self).__init__()

        self.filepath = filepath

        self.swa_epoch = swa_epoch 

    

    def on_train_begin(self, logs=None):

        self.nb_epoch = self.params['epochs']

        print('Stochastic weight averaging selected for last {} epochs.'

              .format(self.nb_epoch - self.swa_epoch))

        

    def on_epoch_end(self, epoch, logs=None):

        

        if epoch == self.swa_epoch:

            self.swa_weights = self.model.get_weights()

            

        elif epoch > self.swa_epoch:    

            for i in range(len(self.swa_weights)):

                self.swa_weights[i] = (self.swa_weights[i] * 

                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)

        else:

            pass

        

    def on_train_end(self, logs=None):

        self.model.set_weights(self.swa_weights)

        print('Final model parameters set to stochastic weight average.')

        self.model.save_weights(self.filepath)

        print('Final stochastic averaged weights saved to file.')
epochs = 50

batchSize = 12



swa = SWA('Best_JointFC_model.h5',epochs - 5)

snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1, init_lr=1e-3)





history = model.fit(X_train, y_train, epochs = epochs, verbose = 1, validation_split = 0.1,

                    validation_steps = 200, steps_per_epoch = len(X_train) //  batchSize,

                    callbacks = snapshot.get_callbacks())
pd.DataFrame(history.history).to_hdf("Model.h5",key="history")
print(f'Present lr: {K.get_value(model.optimizer.lr)}')
loss, accuracy, _, precision, recall = model.evaluate(X_test, y_test, verbose=0)

print(f'Loss: {np.round(loss,4)}\nAccuracy: {100 * np.round(accuracy,4)}%\nPrecision: {np.round(precision,4)}\nRecall: {np.round(recall, 4)}')
y_pred = model.predict(X_test)



predictions = []

testLabels = []



for pred in y_pred:

    predictions.append(np.argmax(pred))

    

for label in y_test:

        testLabels.append(np.where(label == 1)[0][0])



        

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        

cm = confusion_matrix(testLabels, predictions)



fig = plt.figure()

ax = fig.add_subplot(111)

matC = ax.matshow(cm)

fig.colorbar(cax)



ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
print(precision_score(testLabels, predictions, average='micro'))

print(recall_score(testLabels, predictions, labels=[1,2], average='micro'))
print(classification_report(testLabels, predictions));