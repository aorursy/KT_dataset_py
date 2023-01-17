# IMPORT MODULES



import os

import keras

import numpy as np  

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout, Embedding, LSTM

from keras import regularizers, layers, preprocessing

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import tensorflow as tf



print(os.listdir("../input"))
# Load the dataset.npy



DataRaw = np.load('../input/dataset.npy', allow_pickle=True)

print(type(DataRaw))

print(DataRaw.ndim)

DataRaw
# As a dictionary

Datadict = DataRaw[()]

print(Datadict)



# As a dataframe

DataDf = pd.DataFrame.from_dict(Datadict)

print(DataDf.shape)

DataDf
# Mean  / Max / Min column width



DataDf.fillna('').astype(str).apply(lambda x:x.str.len()).max()
# Is the data balanced ?



DataDf.groupby('resistant').size().plot.bar()

plt.show()
# Tokenize from characters to integers (sequences and then pad / truncate data)



Datatok = DataDf.copy()

maxlen = 160 # cut off after this number of characters in a string



max_words = 4 # considers only the top number of characters in the dictionary A C T G

max_features = max_words



tokenizer = Tokenizer(num_words=max_words, char_level=True)

tokenizer.fit_on_texts(list(Datatok['genes']))

sequences = tokenizer.texts_to_sequences(list(Datatok['genes']))

word_index = tokenizer.word_index

Xpad = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post', value=0)



print('Found %s unique tokens.' % len(word_index))

print('word_index', word_index)
# Separate the label



labels = np.asarray(Datatok['resistant'])

print(Xpad.shape)

print(labels.shape)
# Check a sample



rowNum = 37149

print(Datatok['genes'][rowNum])

print(sequences[rowNum])

print(Xpad[rowNum])

print(labels[rowNum])
# Create train & val and test datasets with inital shuffle (as the original dataset may be arranged)



training_samples = int(Xpad.shape[0] * 0.9)

# The validation is being taken by keras - below

# test = remaining



indices = np.arange(Xpad.shape[0])

np.random.shuffle(indices) # FOR TESTING PURPOSES comment it out - to keep indices as above



Xpad = Xpad[indices]

labels = labels[indices]



x_train = Xpad[:training_samples]

y_train = labels[:training_samples]

x_test = Xpad[training_samples: ]

y_test = labels[training_samples: ]



print('x_train', x_train.shape)

print('y_train', y_train.shape)

print('x_test', x_test.shape)

print('y_test', y_test.shape)
# Model ... 128 CNN window 27 & Bidirectional GRU accuracy = 



model = Sequential()

model.add(Embedding(4, 1, input_length=maxlen))

model.add(layers.Conv1D(128, 27, activation='relu'))

model.add(layers.MaxPooling1D(9))

model.add(layers.Dropout(0.5))

model.add(layers.Conv1D(128, 9, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Bidirectional(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()
# Train / Validate model



history = model.fit(x_train, y_train,

epochs = 10,

batch_size=32,

validation_split=0.2)
# Learning curves



# VALIDATION LOSS curves



plt.clf()

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, (len(history_dict['loss']) + 1))

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



# VALIDATION ACCURACY curves



plt.clf()

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

epochs = range(1, (len(history_dict['acc']) + 1))

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
# Final Predict on test



final_predictions = model.predict(x_test)

print(final_predictions)



# Modify the raw final_predictions - prediction probs  - into 0 and 1

# Cutoff point = 0.5



Preds = final_predictions.copy()

print(len(Preds))



Preds[ np.where( Preds >= 0.5 ) ] = 1

Preds[ np.where( Preds < 0.5 ) ] = 0

print(Preds)
# Confusion matrix



conf_mx = confusion_matrix(y_test, Preds)



TN = conf_mx[0,0]

FP = conf_mx[0,1]

FN = conf_mx[1,0]

TP = conf_mx[1,1]



print ('TN: ', TN)

print ('FP: ', FP)

print ('FN: ', FN)

print ('TP: ', TP)



recall = TP/(TP+FN)

precision = TP/(TP+FP)



print (recall, precision)
# Function to visualize the confusion matrix



def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,

                          normalize=False):

    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
plot_confusion_matrix(conf_mx, 

                      normalize    = False,

                      target_names = ['resistant', 'sensistive'],

                      title        = "Confusion Matrix ")
print ('precision ',precision_score(y_test, Preds))

print ('recall ',recall_score(y_test, Preds) )

print ('accuracy ',accuracy_score(y_test, Preds))

print ('F1 score ',f1_score(y_test, Preds))
# AUC/ROC curves should be used when there are roughly equal numbers of observations for each class

# Precision-Recall curves should be used when there is a moderate to large class imbalance



# calculate AUC

auc = roc_auc_score(y_test, Preds)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, Preds)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')

plt.title('ROC ')

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, Preds)

# calculate F1 score

f1 = f1_score(y_test, Preds)

# calculate average precision score

ap = average_precision_score(y_test, Preds)

print('f1=%.3f ap=%.3f' % (f1, ap))

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the roc curve for the model

plt.plot(recall, precision, marker='.')

plt.show()
# From nucleotides to codons ... w/o considering the start / stop codons as the data is synthetic and may not have these



DataCod = DataDf.copy()



Codons = list(DataCod['genes'])

print(len(Codons))



for n in range(len(Codons)):

    Codons[n] = list([Codons[n][i:i+3] for i in range(0, len(Codons[n]), 3)])

    

DataCod['codons'] = Codons

DataCod
# Tokenize from codons to integers (sequences and then pad / truncate data)



maxlen = 53 # cut off after this number of codons in a list



max_words = 64 # considers only the top number of codons  in the dictionary (It finds 66 below because of 'a' and 'ga')

max_features = max_words



#tokenizer = Tokenizer(num_words=max_words, char_level=True)

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(list(DataCod['codons']))

sequences = tokenizer.texts_to_sequences(list(DataCod['codons']))

word_index = tokenizer.word_index

Xpad = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post', value=0)



print('Found %s unique tokens.' % len(word_index))

print('word_index', word_index)
# Separate the label



labels = np.asarray(DataCod['resistant'])

print(Xpad.shape)

print(labels.shape)
# Check a sample



rowNum = 37149

print(DataCod['genes'][rowNum])

print(DataCod['codons'][rowNum])

print(sequences[rowNum])

print(Xpad[rowNum])

print(labels[rowNum])
# Create train & val and test datasets with inital shuffle (as the original dataset may be arranged)



training_samples = int(Xpad.shape[0] * 0.9)

# The validation is being taken by keras - below

# test = remaining



indices = np.arange(Xpad.shape[0])

np.random.shuffle(indices) # FOR TESTING PURPOSES comment it out - to keep indices as above



Xpad = Xpad[indices]

labels = labels[indices]



x_train = Xpad[:training_samples]

y_train = labels[:training_samples]

x_test = Xpad[training_samples: ]

y_test = labels[training_samples: ]



print('x_train', x_train.shape)

print('y_train', y_train.shape)

print('x_test', x_test.shape)

print('y_test', y_test.shape)
# Model ... 64 CNN window 27 & Bidirectional GRU accuracy = 0.99



model = Sequential()

model.add(Embedding(64, 1, input_length=maxlen))

model.add(layers.Conv1D(128, 27, activation='relu'))

model.add(layers.MaxPooling1D(3))

model.add(layers.Dropout(0.5))

model.add(layers.Conv1D(128, 9, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Bidirectional(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()
# Train / Validate model



history = model.fit(x_train, y_train,

epochs = 10,

batch_size=32,

validation_split=0.2)
# Learning curves



# VALIDATION LOSS curves



plt.clf()

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, (len(history_dict['loss']) + 1))

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



# VALIDATION ACCURACY curves



plt.clf()

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

epochs = range(1, (len(history_dict['acc']) + 1))

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
# Final Predict on test



final_predictions = model.predict(x_test)

print(final_predictions)



# Modify the raw final_predictions - prediction probs  - into 0 and 1

# Cutoff point = 0.5



Preds = final_predictions.copy()

print(len(Preds))



Preds[ np.where( Preds >= 0.5 ) ] = 1

Preds[ np.where( Preds < 0.5 ) ] = 0

print(Preds)
# Confusion matrix



conf_mx = confusion_matrix(y_test, Preds)



TN = conf_mx[0,0]

FP = conf_mx[0,1]

FN = conf_mx[1,0]

TP = conf_mx[1,1]



print ('TN: ', TN)

print ('FP: ', FP)

print ('FN: ', FN)

print ('TP: ', TP)



recall = TP/(TP+FN)

precision = TP/(TP+FP)



print (recall, precision)
plot_confusion_matrix(conf_mx, 

                      normalize    = False,

                      target_names = ['resistant', 'sensistive'],

                      title        = "Confusion Matrix ")
# AUC/ROC curves should be used when there are roughly equal numbers of observations for each class

# Precision-Recall curves should be used when there is a moderate to large class imbalance



# calculate AUC

auc = roc_auc_score(y_test, Preds)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, Preds)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')

plt.title('ROC ')

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, Preds)

# calculate F1 score

f1 = f1_score(y_test, Preds)

# calculate average precision score

ap = average_precision_score(y_test, Preds)

print('f1=%.3f ap=%.3f' % (f1, ap))

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the roc curve for the model

plt.plot(recall, precision, marker='.')

plt.show()