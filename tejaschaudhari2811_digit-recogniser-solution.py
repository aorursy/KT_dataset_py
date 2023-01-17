import pandas as pd # pd is alias for pandas
import sys
import matplotlib.pyplot as plt #plt alias for module pyplot
%matplotlib inline 
# to print the results here itself
import numpy as np
from keras.models import Sequential #ANN architecture
from keras.layers import Dense #The layers in ANN
from keras.utils import to_categorical
import mnist
np.set_printoptions(threshold=sys.maxsize)
print("setup complete")

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
print("runs")
#normalize the data from [0,255] to [0,1]
train_images = (train_images/255)
test_images = (test_images/255)
train_images = train_images.reshape(-1,784)
test_images = test_images.reshape(-1,784)

model = Sequential()
model.add(Dense(64, activation ='relu',input_dim=784))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
#compile the model
#It needs and optimizer and loss function. The loss function measures how well the model did on training and trirs to improve using optimizer

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',#(classes that are greater than 2):
    metrics = ['accuracy']
)
#train the model
model.fit(
        train_images,
        to_categorical(train_labels, num_classes=10), # it transforms into 1D array with 10 values
        epochs = 4, #the number of iterations over the entire dataset to train on
        batch_size = 50, #the number of samples per gradient update
        )
#Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)
#predict first 5 images
predictions = model.predict(test_images[:len(test_images)])
print(np.argmax(predictions, axis = 1))
print(test_labels[:len(test_images)])
for i in range(0,5):
    first_image=test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(pixels, cmap="gray")
    plt.show()
