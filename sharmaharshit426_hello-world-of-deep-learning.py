#For getting the mnist handwritten dataset

from keras.datasets import mnist



#We will be using the methods present inside the ImageDataGenerator class to feed our MNIST data to the NN

from keras.preprocessing.image import ImageDataGenerator



#Importing pyplot for data visualization

import matplotlib.pyplot as plt

#Setting matplotlib as inline will display the corresponding graph below the cell itself

%matplotlib inline



#For some fancy Visualizations

import seaborn as sns

sns.set_style('dark')





#To generate random numbers

import random



#Importing DL libraries

from keras.layers import Conv2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import MaxPool2D

from keras.models import Sequential



from keras.utils import np_utils



import numpy as np
#Loading the data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}\t\ty_train.shape: {y_train.shape}")

print(f"x_test shape: {x_test.shape}\t\ty_test.shape: {y_test.shape}")

#Making a pyplot figure

plt.figure()



#subplot(r,c) provide the no. of rows and columns

f, axarr = plt.subplots(4,4, figsize=(8,8)) 



# use the created array to output your multiple images. In this case I have stacked 4 images vertically

for i in range(4):

    for j in range(4):

        num = random.randint(0,500)

        axarr[i][j].imshow(x_train[num], cmap='gray', interpolation='none')

        axarr[i][j].set_title(f'Label: {y_train[num]}')

        plt.tight_layout()

        
#Reshaping X so as to make it compatible with the way Keras accepts Images into the NN.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

#Data Preprocessing to make the mean 0. It speeds up NN training time.

x_train = x_train/255

x_test= x_test/255



#Since there are 10 digits, n_classes = 10

n_classes = 10

print(f"Raw labels: {y_train[0]}")



'''

Since y_train and y_test consist of only digits at this point, we will be converting it to one-hot matrix,

to feed it to the NN

'''

y_train = np_utils.to_categorical(y_train, num_classes=n_classes)

y_test = np_utils.to_categorical(y_test, num_classes=n_classes)



print(f"Example of a one-hot matrix: {y_train[0]}")



print("\nAfter reshaping")

print(f"x_train shape: {x_train.shape}\t\ty_train.shape: {y_train.shape}")

print(f"x_test shape: {x_test.shape}\t\ty_test.shape: {y_test.shape}")
#Building the model

model = Sequential()



model.add(Conv2D(filters=16, kernel_size=3, input_shape=(28, 28, 1), activation='relu', use_bias=True))

model.add(MaxPool2D(2,2))



model.add(Conv2D(filters=16, kernel_size=3,activation='relu', use_bias=True))

model.add(MaxPool2D(2,2))



model.add(Flatten())



model.add(Dense(32, activation='relu'))



model.add(Dense(10, activation='softmax'))



model.summary()

#To reduce the learning rate once the loss reaches a plateau

from keras.callbacks import ReduceLROnPlateau

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics = ['accuracy'])
epochs = 5

history_1 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=1, batch_size=128)


# Plot training & validation loss values

sns.lineplot(x=range(epochs), y=history_1.history['loss'])

sns.lineplot(x=range(epochs), y=history_1.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()





# Plot training & validation accuracy values

sns.lineplot(x=range(epochs), y=history_1.history['accuracy'])

sns.lineplot(x=range(epochs), y=history_1.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_2 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=1, batch_size=128)
# Plot training & validation loss values

sns.lineplot(x=range(epochs), y=history_2.history['loss'])

sns.lineplot(x=range(epochs), y=history_2.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()





# Plot training & validation accuracy values

sns.lineplot(x=range(epochs), y=history_2.history['accuracy'])

sns.lineplot(x=range(epochs), y=history_2.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()

#Let's predict some images from the test set 

fig = plt.figure()

f, axarr = plt.subplots(4,4, figsize=(8, 8))

for i in range(4):

    for j in range(4):

        num = random.randint(0, 10000)

        img = x_test[num].reshape(28, 28)

        img_input = img.reshape(1, 28, 28, 1)

        actual_value = np.argmax(y_test[num])

        output = model.predict(img_input)

        prediction = np.argmax(output)

        axarr[i, j].imshow(img, cmap='gray')

        axarr[i, j].set_title(f"Predicted Value: {prediction}\n Actual Value: {actual_value}")

        plt.tight_layout()