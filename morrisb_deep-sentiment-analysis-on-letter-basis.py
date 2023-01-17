# To do linear algebra
import numpy as np

# To store data
import pandas as pd

# To open zipped files
import bz2

# To use regular expressions
import re

# To shuffle training- & testing-dataset
from sklearn.utils import shuffle

# To build models
from keras.layers import Dense, Input, Dropout, Flatten, Conv1D, BatchNormalization, MaxPool1D
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding

# To get progression bars
from tqdm import tqdm

# To remove rare words
from gensim.corpora import Dictionary

# To search directories
import os

# To create plots
import matplotlib.pyplot as plt

# To aggressively suppress warnings
import warnings
warnings.filterwarnings("ignore")
# Open file and read lines
train_file = bz2.BZ2File('../input/train.ft.txt.bz2')
test_file = bz2.BZ2File('../input/test.ft.txt.bz2')

# Decode bytes
train_lines = [x.decode('utf-8') for x in train_file.readlines()]
test_lines = [x.decode('utf-8') for x in test_file.readlines()]
del train_file, test_file


def createDataset(lines):
    # Create mapping for positive-negative labels
    label_mapping = {'__label__1':[1,0], '__label__2':[0,1]}
    
    y = []
    X = []
    # Iterate over all lines
    for line in lines:
        # Split label and review
        y_tmp, X_tmp = line.split(' ', 1)
    
        # Store labels and texts
        y.append(label_mapping[y_tmp])
        X.append(X_tmp)
    return X, y

# Get train- & testset
X_train_data, y_train_data = createDataset(train_lines)
X_test_data, y_test_data = createDataset(test_lines)
del train_lines, test_lines

print('Length Train:\t{}\nLength Test:\t{}\n'.format(len(y_train_data), len(y_test_data)))
# Display random sample
print('Random Review:\n', X_train_data[np.random.choice(range(len(X_train_data)))])


# All signs in the dataset (can be substituted by iterating over the entire dataset)
signs = '\x02\x03\x04\x05\x07\x08\n\x0f\x10\x12\x13\x14\x15\x16\x17\x18\x19\x1b !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f¡´¿ÀÁÃÄÈÉÌÍÑÒÓÖÙÚÜßàáäèéìíñòóöùúüœƒΩ–—‘’“”„†•…′″€™↓∂∅≠⊂⊕♠♣♥♦⟨'

# Create dictionary mapping from sign to id
dictionary = Dictionary([list(signs)])
print('Different Signs In The Dataset:\t{}'.format(len(dictionary)))


def generatorDictionary(data):
    # Iterate over all reviews
    for review in tqdm(data):
        # Yield tokenized review
        yield list(review)

# Create dictionary by iterating over the entire dataset
#dictionary = Dictionary(generatorDictionary(X_train_data))
# Length of the sequence for the model
maxlen = 800

def generator(X, y, batchsize, length):
    # Iterate over all samples and yield batches
    while True:
        # Shuffle the dataset
        X, y = shuffle(X, y)
    
        # Variables to save the batch
        X_data = []
        y_data = []
        # Iterate over the dataset
        for review, label in zip(X, y):
            # Transform review to sequence of indices
            review_idx = np.array(dictionary.doc2idx(list(review)))
            # Filter unknown indices
            filtered_review_idx = review_idx[review_idx!=-1]
            # Limit sequence to maxlen entries
            pad_review_idx = pad_sequences([filtered_review_idx], maxlen=length, padding='post', value=-1)[0]
            # Append sequence and label to batch
            X_data.append(pad_review_idx)
            y_data.append(label)
        
            # Yield batch if batchsize is reached
            if len(y_data)==batchsize:
                yield (np.array(X_data), np.array(y_data))
                X_data = []
                y_data = []
def createModel():
    # Input layer
    input = Input(shape=[maxlen])

    # Embedding to reduce featurespace
    net = Embedding(len(dictionary), 30)(input)

    # 1D Convolution
    net = BatchNormalization()(net)
    net = Conv1D(16, 7, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv1D(24, 7, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv1D(32, 7, padding='same', activation='relu')(net)
    net = MaxPool1D(pool_size=5)(net)

    # 1D Convolution
    net = BatchNormalization()(net)
    net = Conv1D(64, 7, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv1D(96, 7, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv1D(128, 7, padding='same', activation='relu')(net)
    net = MaxPool1D(pool_size=5)(net)

    # 1D Convolution
    net = BatchNormalization()(net)
    net = Conv1D(256, 7, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv1D(384, 7, padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv1D(512, 7, padding='same', activation='relu')(net)
    net = MaxPool1D(pool_size=5)(net)

    # Flatten
    net = Flatten()(net)

    # Dense layer for combination
    net = BatchNormalization()(net)
    net = Dropout(0.2)(net)
    net = Dense(1024, activation='relu')(net)

    # Dense layer for combination
    net = BatchNormalization()(net)
    net = Dropout(0.2)(net)
    net = Dense(1024, activation='relu')(net)

    # Dense layer for combination
    net = BatchNormalization()(net)
    net = Dropout(0.2)(net)
    output = Dense(2, activation='softmax')(net)


    # Create and compile model
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # Display model
    model.summary()
    
    return model
# Create model
model = createModel()

# Setup batch generators
train_generator = generator(X_train_data, y_train_data, 1000, maxlen)
valid_generator = generator(X_test_data[:300000], y_test_data[:300000], 1000, maxlen)
test_generator = generator(X_test_data[300000:], y_test_data[300000:], 1000, maxlen)
    
# Fit model
model.fit_generator(generator=train_generator, steps_per_epoch=500, epochs=45, validation_data=valid_generator, validation_steps=300)

# Test model
test_loss, test_accuracy = model.evaluate_generator(generator=test_generator, steps=100)
# Get training history
history = model.history.history

# Create subplots
fig, axarr = plt.subplots(2, 1, figsize=(12,8))
# Plot accuracy
axarr[0].plot(history['acc'], label='Train')
axarr[0].plot(history['val_acc'], label='Val')
axarr[0].set_title('{:.4f}: Training Accuracy'.format(test_accuracy))
axarr[0].set_xlabel('Epoch')
axarr[0].set_ylabel('Accuracy')
axarr[0].legend()
# Plot loss
axarr[1].plot(history['loss'], label='Train')
axarr[1].plot(history['val_loss'], label='Val')
axarr[1].set_title('{:.4f}: Training Losses'.format(test_loss))
axarr[1].set_xlabel('Epoch')
axarr[1].set_ylabel('Loss')
axarr[1].legend()

plt.tight_layout()
plt.show()
