from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('Training data: ', train_images.shape, train_labels.shape)

print('Testing data: ', test_images.shape, test_labels.shape)
print(train_labels[0:20])
#Unique numbers from the train labels
total_classes = np.unique(train_labels)
number_of_total_classes = len(total_classes)
print('Total number of output classes : ', number_of_total_classes)
print('All Output classes : ', total_classes)
# Plotting some sample data
plt.figure(figsize=[10,5])
# Displaying the first image in training data
plt.subplot(121)
plt.imshow(train_images[0,:,:], cmap='gray')
plt.title("Label : {}".format(train_labels[0]))

# Displaying the first image in testing data
plt.figure(figsize=[10,5])
plt.subplot(122)
plt.imshow(test_images[0,:,:], cmap='gray')
plt.title("Label : {}".format(test_labels[0]))
# Change from matrix to array of dimension 28x28 to array of dimention 784
newdimnsionData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], newdimnsionData)
test_data = test_images.reshape(test_images.shape[0], newdimnsionData)
print(train_data.shape)
print(test_data.shape)
# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255
# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label 6 : ', train_labels[10])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[10])
total_classes
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(newdimnsionData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(number_of_total_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result: Loss = {}, accuracy = {}".format(test_loss, test_acc))
