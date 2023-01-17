# Imports for this block 
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

# Load training data into an ndarray (skip the col headers - the first line in the csv file)
train = np.genfromtxt("../input/training/train.csv", skip_header=1, delimiter=",")

# Separate labels from data (for the train file): Labels are the first column
Y_train = train[:,0:1]
X_train = train[:,1:]
del train

# normalize integers (0-255) into floats (0-1)
X_train = X_train/255.0

# Reshape the ndarray from 42000,784,1 -> 42000,28,28,1 (the ,1 is the channel number - needed for conv nets)
X_train = X_train.reshape([-1,28,28,1])

# Encode labels to one hot vectors (eg : 2 -> [0,0,1,0,0,0,0,0,0,0] etc)
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the training set (using sklearn utility)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print('X_train shape = ', X_train.shape)
print('Y_train shape = ', Y_train.shape)
print('X_val shape = ', X_val.shape)
print('Y_val shape = ', Y_val.shape)

# New imports for this block
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Build a quick conv net
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Let's see what the layers look like
print(model.summary())

# New imports for this block
from keras.optimizers import RMSprop

# Compile the model
model.compile(optimizer=RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

# Fit the model (running the validation test as we go and recording loss and accuracy on 'unseen' images)
history = model.fit(X_train, Y_train, epochs=30, batch_size=100, validation_data=(X_val, Y_val))

# New imports for this block
import matplotlib.pyplot as plt

# Get the loss and accuracy info from the history history dictionary
training_losses = history.history['loss']
validation_losses = history.history['val_loss']
training_acc = history.history['acc']
validation_acc = history.history['val_acc']

# label epochs from 1 not 0
epochs = range(1, 31)

plt.plot(epochs, training_losses, 'bo', label='Training loss')
plt.plot(epochs, validation_losses, 'b', label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, training_acc, 'bo', label='Training accuracy')
plt.plot(epochs, validation_acc, 'b', label='Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Save the net we have
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Load training data into an ndarray (skip the col headers - the first line in the csv file)
test = np.genfromtxt("../input/testing/test.csv", skip_header=1, delimiter=",")

# Normalize and reshape the data into 28x28x1 tensors
test = test/255.0
test = test.reshape([-1,28,28,1])

# Make a ndarray of digit probabilities
predictions = model.predict(test)

# Find the most probable digit in each row of the array
digit_predictions = np.argmax(predictions, axis=1)

# Write to file etc...

print('Done')

