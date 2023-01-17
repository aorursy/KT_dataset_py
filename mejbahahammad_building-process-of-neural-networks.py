from sklearn import preprocessing

import numpy as np

from keras import models

from keras import layers

from keras.datasets import reuters

from keras.utils.np_utils import to_categorical

from keras.datasets import imdb

from keras.preprocessing.text import Tokenizer

# Create feature

features = np.array([[-100.1, 3240.1],

[-200.2, -234.1],

[5000.5, 150.1],

[6000.6, -125.1],

[9000.9, -673.1]])
scaler = preprocessing.StandardScaler()

# Transform the feature

features_standardized = scaler.fit_transform(features)

# Show feature

features_standardized
# Print mean and standard deviation

print("Mean:", round(features_standardized[:,0].mean()))

print("Standard deviation: ", features_standardized[:,0].std())
# Start neural network

network = models.Sequential()
# Add fully connected layer with a ReLU activation function

network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))
# Add fully connected layer with a ReLU activation function

network.add(layers.Dense(units=16, activation="relu"))
# Add fully connected layer with a sigmoid activation function

network.add(layers.Dense(units=1, activation="sigmoid"))
network.compile(loss="binary_crossentropy", # Cross-entropy

optimizer="rmsprop", # Root Mean Square Propagation

metrics=["accuracy"]) # Accuracy performance metric
np.random.seed(0)

# Set the number of features we want

number_of_features = 1000
# Load data and target vector from movie review data

(data_train, target_train), (data_test, target_test) = imdb.load_data(

num_words=number_of_features)
# Convert movie review data to one-hot encoded feature matrix

tokenizer = Tokenizer(num_words=number_of_features)

features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")

features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Start neural network

network = models.Sequential()

# Add fully connected layer with a ReLU activation function

network.add(layers.Dense(units=16, activation="relu", input_shape=(

number_of_features,)))
# Add fully connected layer with a ReLU activation function

network.add(layers.Dense(units=16, activation="relu"))

# Add fully connected layer with a sigmoid activation function

network.add(layers.Dense(units=1, activation="sigmoid"))
network.compile(loss="binary_crossentropy", # Cross-entropy

optimizer="rmsprop", # Root Mean Square Propagation

metrics=["accuracy"]) # Accuracy performance metric
#Train neural network

history = network.fit(features_train, # Features

target_train, # Target vector

epochs=3, # Number of epochs

verbose=1, # Print description after each epoch

batch_size=100, # Number of observations per batch

validation_data=(features_test, target_test)) # Test data
# View shape of feature matrix

features_train.shape
# Set random seed

np.random.seed(0)

# Set the number of features we want

number_of_features = 5000
data = reuters.load_data(num_words=number_of_features)

(data_train, target_vector_train), (data_test, target_vector_test) = data

# Convert feature data to a one-hot encoded feature matrix

tokenizer = Tokenizer(num_words=number_of_features)

features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")

features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# One-hot encode target vector to create a target matrix

target_train = to_categorical(target_vector_train)

target_test = to_categorical(target_vector_test)

# Start neural network

network = models.Sequential()
# Add fully connected layer with a ReLU activation function

network.add(layers.Dense(units=100,

activation="relu",

input_shape=(number_of_features,)))
# Add fully connected layer with a ReLU activation function

network.add(layers.Dense(units=100, activation="relu"))

# Add fully connected layer with a softmax activation function

network.add(layers.Dense(units=46, activation="softmax"))
#Compile neural network

network.compile(loss="categorical_crossentropy", # Cross-entropy

optimizer="rmsprop", # Root Mean Square Propagation

metrics=["accuracy"]) # Accuracy performance metric
# Train neural network

history = network.fit(features_train, # Features

target_train, # Target

epochs=3, # Three epochs

verbose=0, # No output

batch_size=100, # Number of observations per batch

validation_data=(features_test, target_test)) # Test data
# View target matrix

target_train