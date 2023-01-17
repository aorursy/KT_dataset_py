import numpy as np
import pandas as pd
# Set seed for reproducability 
np.random.seed(42)
import matplotlib.pyplot as plt
df = pd.read_csv('../input/processed_bank.csv',index_col=0)
# Display first few (5) rows
df.head()
# Check shape of dataset
df.shape
# X is everything that is not y
X = df.loc[:, df.columns != 'y'].values
# y is y
y = df['y'].values
# First split in train / test_dev
from sklearn.model_selection import train_test_split
X_train, X_test_dev, y_train, y_test_dev = train_test_split(X, y, test_size=0.25, random_state=0)

# Second split in dev / test
X_dev, X_test, y_dev, y_test = train_test_split(X_test_dev, y_test_dev, test_size=0.5, random_state=0)

# Remove test_dev set from memory
del X_test_dev
del y_test_dev
# Get Keras
# We will build a simple sequential model
from keras.models import Sequential
# Using fully connected layers
from keras.layers import Dense
# With vanilla gradient descent
from keras.optimizers import SGD
# Sequential model
model = Sequential()

# Logistic regresison is a single layer network
model.add(Dense(1,activation='sigmoid',input_dim=64))

# Compile the model
model.compile(optimizer=SGD(lr=0.01),loss='binary_crossentropy',metrics=['acc'])
history_log = model.fit(X_train, y_train, epochs=1000, batch_size=X_train.shape[0], verbose=0)
plt.style.use('fivethirtyeight')
plt.plot(history_log.history['acc'], label = 'Logistic regression')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_log.history['loss'], label='Logistic regression')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
model.evaluate(x=X_dev,y=y_dev)
# Sequential model
model = Sequential()

# First hidden layer
model.add(Dense(32,activation='tanh',input_dim=64))

# Second hidden layer
model.add(Dense(16,activation='tanh'))

# Output layer
model.add(Dense(1,activation='sigmoid'))

# Compile the model
model.compile(optimizer=SGD(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])


# Train
history_tanh = model.fit(X_train, y_train, # Train on training set
                         epochs=1000, # We will train over 1,000 epochs
                         batch_size=X_train.shape[0], # Batch size = training set size
                         verbose=0) # Suppress Keras output
plt.plot(history_log.history['acc'], label= 'Logistic Regeression')
plt.plot(history_tanh.history['acc'], label= '2 hidden layer Tanh')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_log.history['loss'], label= 'Logistic Regeression')
plt.plot(history_tanh.history['loss'], label= '2 hidden layer Tanh')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
model.evaluate(x=X_dev,y=y_dev)
# Sequential model
model = Sequential()

# First hidden layer now with relu!
model.add(Dense(32,activation='relu',input_dim=64))

# Second hidden layer now with relu!
model.add(Dense(16,activation='relu'))

# Output layer stayed sigmoid
model.add(Dense(1,activation='sigmoid'))

# Compile the model
model.compile(optimizer=SGD(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])

# Train
history_relu = model.fit(X_train, y_train, # Train on training set
                         epochs=1000, # We will train over 1,000 epochs
                         batch_size=X_train.shape[0], # Batch size = training set size
                         verbose=0) # Suppress Keras output
plt.plot(history_tanh.history['acc'], label='2 hidden layer Tanh')
plt.plot(history_relu.history['acc'], label='2 hidden layer ReLu')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_tanh.history['loss'], label='2 hidden layer Tanh')
plt.plot(history_relu.history['loss'], label='2 hidden layer ReLu')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Sequential model
model = Sequential()

# First hidden layer
model.add(Dense(32,activation='relu',input_dim=64))

# Second hidden layer
model.add(Dense(16,activation='relu'))

# Output layer
model.add(Dense(1,activation='sigmoid'))

# Setup optimizer with learning rate of 0.01 and momentum (beta) of 0.9
momentum_optimizer = SGD(lr=0.01, momentum=0.9)

# Compile the model
model.compile(optimizer=momentum_optimizer,
              loss='binary_crossentropy',
              metrics=['acc'])

# Train
history_momentum = model.fit(X_train, y_train, # Train on training set
                             epochs=1000, # We will train over 1,000 epochs
                             batch_size=X_train.shape[0], # Batch size = training set size
                             verbose=0) # Suppress Keras output
plt.plot(history_relu.history['acc'], label= '2 hidden layer ReLu')
plt.plot(history_momentum.history['acc'], label= '2 hidden layer ReLu + Momentum')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_relu.history['loss'], label= '2 hidden layer ReLu')
plt.plot(history_momentum.history['loss'], label= '2 hidden layer ReLu + Momentum')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
model.evaluate(x=X_dev,y=y_dev)
from keras.optimizers import adam
# Sequential model
model = Sequential()

# First hidden layer
model.add(Dense(32,activation='relu',input_dim=64))

# Second hidden layer
model.add(Dense(16,activation='relu'))

# Output layer stayed sigmoid
model.add(Dense(1,activation='sigmoid'))

# Setup adam optimizer
adam_optimizer=adam(lr=0.1,
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08)

# Compile the model
model.compile(optimizer=adam_optimizer,
              loss='binary_crossentropy',
              metrics=['acc'])


# Train
history_adam = model.fit(X_train, y_train, # Train on training set
                         epochs=1000, # We will train over 1,000 epochs
                         batch_size=X_train.shape[0], # Batch size = training set size
                         verbose=0) # Suppress Keras output
plt.plot(history_relu.history['acc'], label = 'Vanilla Gradient Descent')
plt.plot(history_momentum.history['acc'], label = 'Momentum Gradient Descent')
plt.plot(history_adam.history['acc'], label = 'Adam')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_relu.history['loss'], label = 'Vanilla Gradient Descent')
plt.plot(history_momentum.history['loss'], label = 'Momentum Gradient Descent')
plt.plot(history_adam.history['loss'], label = 'Adam')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
model.evaluate(x=X_dev,y=y_dev)
from keras import regularizers
# Sequential model
model = Sequential()

# First hidden layer now regularized
model.add(Dense(32,activation='relu',
                input_dim=64,
                kernel_regularizer = regularizers.l2(0.01)))

# Second hidden layer now regularized
model.add(Dense(16,activation='relu',
                   kernel_regularizer = regularizers.l2(0.01)))

# Output layer stayed sigmoid
model.add(Dense(1,activation='sigmoid'))

# Setup adam optimizer
adam_optimizer=adam(lr=0.1,
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08)

# Compile the model
model.compile(optimizer=adam_optimizer,
              loss='binary_crossentropy',
              metrics=['acc'])

# Train
history_regularized=model.fit(X_train, y_train, # Train on training set
                             epochs=1000, # We will train over 1,000 epochs
                             batch_size=X_train.shape[0], # Batch size = training set size
                             verbose=0) # Suppress Keras output
plt.plot(history_adam.history['acc'], label = 'No Regularization')
plt.plot(history_regularized.history['acc'], label = 'Regularization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_adam.history['loss'], label = 'No Regularization')
plt.plot(history_regularized.history['loss'], label = 'Regularization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
model.evaluate(x=X_dev,y=y_dev)
from keras.layers import Dropout
# Sequential model
model = Sequential()

# First hidden layer
model.add(Dense(32,activation='relu',
                input_dim=64))

# Add dropout layer
model.add(Dropout(rate=0.5))

# Second hidden layer
model.add(Dense(16,activation='relu'))


# Add another dropout layer
model.add(Dropout(rate=0.5))

# Output layer stayed sigmoid
model.add(Dense(1,activation='sigmoid'))

# Setup adam optimizer
adam_optimizer=adam(lr=0.1,
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08)

# Compile the model
model.compile(optimizer=adam_optimizer,
              loss='binary_crossentropy',
              metrics=['acc'])

# Train
history_dropout = model.fit(X_train, y_train, # Train on training set
                             epochs=1000, # We will train over 1,000 epochs
                             batch_size=X_train.shape[0], # Batch size = training set size
                             verbose=0) # Suppress Keras output
plt.plot(history_adam.history['acc'], label = 'No Regularization')
plt.plot(history_regularized.history['acc'], label = 'L2 Regularization')
plt.plot(history_dropout.history['acc'], label = 'Dropout Regularization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_adam.history['loss'], label = 'No Regularization')
plt.plot(history_regularized.history['loss'], label = 'L2 Regularization')
plt.plot(history_dropout.history['loss'], label = 'Dropout Regularization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
model.evaluate(x=X_dev,y=y_dev)
from keras.layers import BatchNormalization
from keras.layers import Activation
# Sequential model
model = Sequential()

# Input layer linear step
model.add(Dense(32, input_dim=64))

# Input layer normalization
model.add(BatchNormalization())

# Input layer activation
model.add(Activation('relu'))

# Add dropout layer
model.add(Dropout(rate=0.5))

# hidden layer linear step
model.add(Dense(16))

# Hidden layer normalization
model.add(BatchNormalization())

# Hidden layer activation
model.add(Activation('relu'))

# Add another dropout layer
model.add(Dropout(rate=0.5))

# Output layer, 
model.add(Dense(1))

# Output normalization
model.add(BatchNormalization())

# Output layer activation
model.add(Activation('sigmoid'))

# Setup adam optimizer
adam_optimizer=adam(lr=0.1,
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08)

# Compile the model
model.compile(optimizer=adam_optimizer,
              loss='binary_crossentropy',
              metrics=['acc'])


# Train
history_batchnorm = model.fit(X_train, y_train, # Train on training set
                             epochs=1000, # We will train over 1,000 epochs
                             batch_size=X_train.shape[0], # Batch size = training set size
                             verbose=0) # Suppress Keras output
plt.plot(history_dropout.history['acc'], label = 'Dropout')
plt.plot(history_batchnorm.history['acc'], label = 'Dropout + Batchnorm')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history_dropout.history['loss'], label = 'Dropout')
plt.plot(history_batchnorm.history['loss'], label = 'Dropout + Batchnorm')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
model.evaluate(x=X_dev,y=y_dev)