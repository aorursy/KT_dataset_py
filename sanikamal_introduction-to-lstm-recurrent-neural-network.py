# Load libraries
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras import layers

# Set random seed
np.random.seed(0)
# import os
# print(os.listdir("../input"))
# Set the number of features we want
number_of_features = 1000

# Load data and target vector from movie review data
(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)
# Use padding or truncation to make each observation have 400 features
train_features = sequence.pad_sequences(train_data, maxlen=400)
test_features = sequence.pad_sequences(test_data, maxlen=400)
# View first observation
print(train_data[0])
# Start neural network
network = models.Sequential()

# Add an embedding layer
network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))

# Add a long short-term memory layer with 128 units
network.add(layers.LSTM(units=128))

# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation='sigmoid'))
# Compile neural network
network.compile(loss='binary_crossentropy', # Cross-entropy
                optimizer='Adam', # Adam optimization
                metrics=['accuracy']) # Accuracy performance metric
network.summary()
# Train neural network
history = network.fit(train_features, # Features
                      train_target, # Target
                      epochs=3, # Number of epochs
                      verbose=1, # Do not print description after each epoch
                      batch_size=1000, # Number of observations per batch
                      validation_data=(test_features, test_target)) # Data for evaluation

