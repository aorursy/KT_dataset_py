# To do linear algebra
import numpy as np

# To store data
import pandas as pd

# To walk directories
import os

# To create models
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, BatchNormalization, Flatten, MaxPool1D
from keras.utils import to_categorical

# To read strings
import json

# To create a submission
import csv

# To create plots
import matplotlib.pyplot as plt
# Path to the data
path = '../input/train_simplified/'

# Get all data files
files = [os.path.join(path, file) for i, file in enumerate(os.listdir(path))]

# Get number of categories
n_categories = len(files)

# Get dictionary to map labels to integers
word_mapping = {file.split('/')[-1][:-4]:i for i, file in enumerate(files)}

print('Number of different files/categories:\t{}'.format(n_categories))
def imageGenerator(batchsize, validation=False):
    # Never ending iterator
    while True:
        
        # Variable to store the data
        df = []
        
        # Iterate over all files
        for file in files:
            # Get random samples of the data
            if validation:
                # Use rows 100000:110000 as validation data
                df.append(pd.read_csv(file, nrows=110000, usecols=[1, 5]).tail(10000).sample(1000))
            else:
                # Use rows :100000 as training data
                df.append(pd.read_csv(file, nrows=100000, usecols=[1, 5]).sample(1000))
                
        # Combine DataFrames
        df = pd.concat(df)
        
        # Use mapping on labels
        df['word'] = df['word'].map(word_mapping)
        
        # Shuffle DataFrame
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Convert labels to vectors
        y = to_categorical(df['word'].values, n_categories)
        
        # Variable to store the sequence of strokes
        X = []
        # Iterate over all images
        for values in df['drawing'].values:
            # Convert string to list
            image = json.loads(values)

            strokes = []
            # Concatenate all strokes
            for x_axis, y_axis in image:
                strokes.extend(list(zip(x_axis, y_axis)))
            strokes = np.array(strokes)

            # Create empty array for sequence padding
            pad = np.zeros((sequence_length, 2))
            
            # Pad/slice data to correct format
            if sequence_length>strokes.shape[0]:
                pad[:strokes.shape[0],:] = strokes
            else:
                pad = strokes[:sequence_length, :]

            X.append(pad)
        X = np.array(X)
        
        
        i = 0
        # Iterate over all batches in the loaded data
        while True:
            # Slice a batch of data to yield
            if i+batchsize<=y.shape[0]:
                y_yield = y[i:i+batchsize]
                X_yield = X[i:i+batchsize]
                i += batchsize
                yield (X_yield, y_yield)
            else:
                break
def createNetwork(seq_len):
    
    # Function to add a convolution layer with batch normalization
    def addConv(network, features, kernel):
        network = BatchNormalization()(network)
        return Conv1D(features, kernel, padding='same', activation='relu')(network)
    
    # Function to add a dense layer with batch normalization and dropout
    def addDense(network, size):
        network = BatchNormalization()(network)
        network = Dropout(0.2)(network)
        return Dense(size, activation='relu')(network)
    
    
    # Input layer
    input = Input(shape=(seq_len, 2))
    network = input
    
    # Add 1D Convolution
    for features in [16, 24, 32]:
        network = addConv(network, features, 5)
    network = MaxPool1D(pool_size=5)(network)
    
    # Add 1D Convolution
    for features in [64, 96, 128]:
        network = addConv(network, features, 5)
    network = MaxPool1D(pool_size=5)(network)

    # Add 1D Convolution
    for features in [256, 384, 512]:
        network = addConv(network, features, 5)
    #network = MaxPool1D(pool_size=5)(network)

    # Flatten
    network = Flatten()(network)
    
    # Dense layer for combination
    for size in [128, 128]:
        network = addDense(network, size)
    
    # Output layer
    output = Dense(len(files), activation='softmax')(network)


    # Create and compile model
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Display model
    model.summary()
    return model


# Length of the x-y-sequences
sequence_length = 80

# Create a model
model = createNetwork(sequence_length)
# Instantiate generators for training and validation data
train_generator = imageGenerator(batchsize=1000)
valid_generator = imageGenerator(batchsize=1000, validation=True)

# Train the model
model.fit_generator(train_generator, steps_per_epoch=340, epochs=130, validation_data=valid_generator, validation_steps=5)
# Get training history
history = model.history.history

print('Final Validation Accuracy: {:.4f}'.format(history['val_acc'][-1]))

# Create subplots
fig, axarr = plt.subplots(2, 1, figsize=(12,9))

# Plot accuracy
axarr[0].plot(history['acc'], label='Train')
axarr[0].plot(history['val_acc'], label='Valid')
axarr[0].set_title('Accuracy')
axarr[0].set_xlabel('Epochs')
axarr[0].set_ylabel('Accuracy')
axarr[0].legend()

# Plot loss
axarr[1].plot(history['loss'], label='Train')
axarr[1].plot(history['val_loss'], label='Valid')
axarr[1].set_title('Loss')
axarr[1].set_xlabel('Epochs')
axarr[1].set_ylabel('Loss')
axarr[1].legend()

plt.tight_layout()
plt.show()
# Load submission data
submission_data = pd.read_csv('../input/test_simplified.csv')


# Preprocess the submission data
X = []
# Iterate over all images
for values in submission_data['drawing'].values:
    # Convert string to list
    image = json.loads(values)
    
    strokes = []
    # Concatenate all strokes
    for x_axis, y_axis in image:
        strokes.extend(list(zip(x_axis, y_axis)))
    strokes = np.array(strokes)

    # Create empty array for padding
    pad = np.zeros((sequence_length, 2))
    # Pad/slice data to correct format
    if sequence_length>strokes.shape[0]:
        pad[:strokes.shape[0],:] = strokes
    else:
        pad = strokes[:sequence_length, :]

    X.append(pad)
X = np.array(X)


# Predict categories
prediction = model.predict(X)

# Slice most probable three categories
best_prediction = np.flip(np.argsort(prediction, axis=1), axis=1)[:, :3]

# Create reverse mapping dictionary
reverse_word_mapping = {word:key for key, word in word_mapping.items()}


submission = []
# Iterate over each image
for key_id, label_ids in zip(submission_data['key_id'].values, best_prediction):
    # reverse map and combine the most probable three categories for each image
    labels = ' '.join([reverse_word_mapping[label_id].replace(' ', '_') for label_id in label_ids])
    submission.append([key_id, labels])

# Create and same submission
pd.DataFrame(submission, columns=['key_id', 'word']).to_csv('Submission.csv', index=False)