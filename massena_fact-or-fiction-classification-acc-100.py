#   Author: Ibrahim Alhas

#   MODEL 3:    CNN with built-in tensorflow tokenizer.
#   This is the final version of the model (not the base).

#   Packages and libraries used for this model.
#   ** Install these if not installed already **.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from time import time
import re
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, \
    classification_report
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split, cross_validate
import tensorflow as tf
import seaborn as sns
import warnings
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D

warnings.filterwarnings('ignore')
# plt.style.use('ggplot')

#   Basic data visualisation and analysis ------------------------------------------------------------------------------
#   We see that the title column is from news articles, and the text column forms the twitter tweet extracts.
true = pd.read_csv('True.csv')
false = pd.read_csv('Fake.csv')

#   We drop the columns we do not need. See chapter 3, model CNN for more details.
true = true.drop('title', axis=1)
true = true.drop('subject', axis=1)
true = true.drop('date', axis=1)
false = false.drop('title', axis=1)
false = false.drop('subject', axis=1)
false = false.drop('date', axis=1)

#   We set the labels for each data instance, where factual = 1, otherwise 0.
false['label'] = 0
true['label'] = 1

#   We merge the two divided datasets (true and fake) into a singular dataset.
data = pd.concat([true, false], ignore_index=True)
texts = data['text']
labels = data['label']
x = texts
y = labels

#   We incorporate the publishers feature from title and text instances, and place it into the dataset manually.
#   First Creating list of index that do not have publication part. We can use this as a new feature.
unknown_publishers = []
for index, row in enumerate(true.text.values):
    try:
        record = row.split(" -", maxsplit=1)
        # if no text part is present, following will give error
        print(record[1])
        # if len of piblication part is greater than 260
        # following will give error, ensuring no text having "-" in between is counted
        assert (len(record[0]) < 260)
    except:
        unknown_publishers.append(index)

#   We print the instances where publication information is absent or different.
print(true.iloc[unknown_publishers].text)

#   We want to use the publication information as a new feature.
publisher = []
tmp_text = []
for index, row in enumerate(true.text.values):
    if index in unknown_publishers:
        #   Append unknown publisher:
        tmp_text.append(row)
        publisher.append("Unknown")
        continue
    record = row.split(" -", maxsplit=1)
    publisher.append(record[0])
    tmp_text.append(record[1])

#   Replace text column with new text + add a new feature column called publisher/source.
true["publisher"] = publisher
true["text"] = tmp_text
del publisher, tmp_text, record, unknown_publishers

#   Validate that the publisher/source column has been added to the dataset.
print(true.head())

#   Check for missing values, then drop them for both datasets.
print([index for index, text in enumerate(true.text.values) if str(text).strip() == ''])
true = true.drop(8970, axis=0)
fakeEmptyIndex = [index for index, text in enumerate(false.text.values) if str(text).strip() == '']
print(f"No of empty rows: {len(fakeEmptyIndex)}")
false.iloc[fakeEmptyIndex].tail()
# -
#   For CNNs, we have to vectorize the text into 2d integers (tensors).
MAX_SEQUENCE_LENGTH = 5000
MAX_NUM_WORDS = 25000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.2
epochs = 1

#   We tokenize the text, just like all other models--------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

#   Print the total number of tokens:
print('Found %s tokens.' % len(word_index))

#   We partition our dataset into train/test.
x_train, x_val, y_train, y_val = train_test_split(data, labels.apply(lambda x: 0 if x == 0 else 1),
                                                  test_size=TEST_SPLIT)
log_dir = "logs\\model\\"
#   A custom callbacks function, which initially included tensorboard.
mycallbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),  # Restoring the best
    #   ...weights will help keep the optimal weights.
    #   tf.keras.callbacks.TensorBoard(log_dir="./logs"),  # NEWLY ADDED - CHECK.
    #   tf.keras.callbacks.TensorBoard(log_dir=log_dir.format(time())),  # NEWLY ADDED - CHECK.
    #   tensorboard --logdir logs --> to check tensorboard feedback.
]

#   Parameters for our model. We experimented with some combinations and settled on this configuration------------------
model = Sequential(
    [
        #   Word/sequence processing:
        layers.Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True),
        #   The layers:
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        #   We classify our model here:
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

#   We compile our model and run, with the loss function crossentropy, and optimizer rmsprop (we experimented with adam,
#   ...but rmsprop produced better results).
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

print("Model weights:")
print(model.weights)

# tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
history = model.fit(x_train, y_train, batch_size=256, epochs=epochs, validation_data=(x_val, y_val),
                    callbacks=mycallbacks)

#   Produce a figure, for every epoch, and show performance metrics.
epochs = [i for i in range(1)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20, 10)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'go-', label='Training Loss')
ax[1].plot(epochs, val_loss, 'ro-', label='Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

'''
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

plt.figure(figsize=(12, 9))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend(prop={'size': 20})
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Accuracy', size=20)
plt.legend(prop={'size': 20})
plt.ylim((0.5, 1))
plt.show()
'''
#   We evaluate our model by predicting a few instances from our test data (the first 5)--------------------------------
print("Evaluation:")
print(model.evaluate(x_val, y_val))

#   We predict a few instances (up to 5).
pred = model.predict(x_val)
print(pred[:5])

binary_predictions = []
for i in pred:
    if i >= 0.5:
        binary_predictions.append(1)
    else:
        binary_predictions.append(0)

#   We print performance metrics:
print('Accuracy on test set:', accuracy_score(binary_predictions, y_val))
print('Precision on test set:', precision_score(binary_predictions, y_val))
print('Recall on test set:', recall_score(binary_predictions, y_val))
print('F1 on test set:', f1_score(binary_predictions, y_val))

#   We print the classification report (as an extra):
print(classification_report(y_val, pred.round(), target_names=['Fact', 'Fiction']))

#   We print the confusion matrix.
cmm = confusion_matrix(y_val, pred.round())
print(cmm)

print("Ibrahim Alhas")

cmm = pd.DataFrame(cmm, index=['Fake', 'Original'], columns=['Fake', 'Original'])
plt.figure(figsize=(10, 10))
sns.heatmap(cmm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['Fake', 'Original'],
            yticklabels=['Fake', 'Original'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#   End----------------------------------------------------