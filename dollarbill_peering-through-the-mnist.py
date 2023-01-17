# For data import
import pandas as pd

# For data manipulation
import numpy as np

# For visualisation
import matplotlib.pyplot as plt
%matplotlib inline

# The real stuff...
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
# Import data
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# Extract data into arrays suitable for training model
training_labels = training_data.loc[:,'label'].values # A numpy array of decimal labels e.g. '4', '7'
training_outputs = np.array([to_categorical(example, num_classes=10) for example in training_labels]) # An array of 10-element arrays, each encoding a label.
training_inputs = np.array(training_data.iloc[:,1:])/255.0 # An array of 784-element arrays, encoding a 28x28 array of pixels. We normalise so that all entries are in the range [0,1].
test_inputs = np.array(test_data)/255.0
# Have a quick look at the data...
for i in range(5):
    plt.subplot(151 + i)
    imgplot = plt.imshow(training_inputs[i].reshape(28,28))
    plt.title(training_labels[i])
    imgplot.set_cmap('binary')
# Set up network topology
network = Sequential()
network.add(Dense(units=100, activation='sigmoid', input_dim=784))
network.add(Dense(units=10, activation='sigmoid'))
# Set up training parameters
network.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=3.0), metrics=['accuracy'])
# Train network
training_history = network.fit(training_inputs, training_outputs, epochs=30, batch_size=10)
# Visualise categorisation accuracy over course of training
plt.plot(training_history.history['acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
# Visualise cost function over course of training
plt.plot(training_history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('cost function')
predicted_labels = [np.argmax(pred) for pred in network.predict(test_inputs)]
predictions = pd.DataFrame({"ImageId": list(range(1,len(predicted_labels)+1)), "Label": predicted_labels})
predictions.to_csv('kaggle_mnist.csv', index=False, header=True)
