import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# loading data
seed = 6
np.random.seed(seed)

training_set = pd.read_csv('../input/fashion-mnist_train.csv')
test_set = pd.read_csv('../input/fashion-mnist_test.csv')
# checking the characteristics of data
training_set.head()
# storing the class labels and image values into separate variables.
X = training_set.iloc[:, 1:].values
y = training_set.iloc[:, 0].values
print ('Input vectors : {}'.format(X.shape))
print ('Class Labels : {}'.format(y.shape))
# reshaping

X = X.reshape(-1,28,28,1)
X.shape
# checking the first 9 images

pixels_x = 28
pixels_y = 28

for i in range(0,9):
    plt.subplot(330 + 1 + i)
    img = X[i].reshape(pixels_x, pixels_y)
    plt.imshow(img, cmap = 'gray')

plt.show()
# splitting the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state = 0)
# normalizing the pixel values
X_train = X_train.astype('float32')/255.0
X_validation = X_validation.astype('float32')/255.0
# checking the class labels
print (y_train[0])
print (y_validation[0])
# converting the numeric class labels to binary form with one hot encoding

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)

y_train[0]
from keras.models import Sequential 
from keras.layers import Conv2D  
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import Dropout
model = Sequential()

# first convolutional and max pooling layers
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# second convolutional and max pooling layers
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())  

model.add(Dense(128, activation='relu'))     # fully connected layer
model.add(Dense(10, activation='softmax'))  # output layer

# Compiling CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()
# fitting the model to training data

history = model.fit(X_train, y_train, validation_data = (X_validation, y_validation), batch_size = 100, epochs = 10)
# evaluating the performance of model on validation set
scores = model.evaluate(X_validation, y_validation, verbose = 1)
print ('Accuracy : {}'.format(scores[1]))
X_test = test_set.iloc[:, 1:].values
y_test = test_set.iloc[:, 0].values

X_test = X_test.reshape(-1, 28, 28, 1)
X_test = X_test.astype('float32')/255.0

X_test.shape
# predicting and storing the values in y_pred
y_pred = model.predict(X_test)
# selecting the class with highest probability
y_pred = np.argmax(y_pred, axis = 1)

from sklearn.metrics import accuracy_score
print ('Accuracy on Test Set = {:0.2f}'.format(accuracy_score(y_test, y_pred)))
