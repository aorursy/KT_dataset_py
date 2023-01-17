#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#Loading data

training = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
testing = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#Dividing data and corresponding label into x_train and y_train
trainingLabels = training['label']
trainingData = training.drop(labels=['label'], axis=1)

#Converting data to numpy array
trainingLabels, trainingData = trainingLabels.values, trainingData.values
#Normalizing Data

trainingData = trainingData / 255.0
testing = testing / 255.0
#Reshape data to image in 3 dimensions
trainingData = trainingData.reshape(-1, 28, 28, 1)
testing = testing.values.reshape(-1, 28, 28, 1)
#One Hot Encoding
trainingLabels = to_categorical(trainingLabels, num_classes = 10)
#Splitting data into training and validation datasets (9:1)
x_train, x_val, y_train, y_val = train_test_split(trainingData, trainingLabels, test_size = 0.1, random_state=2)
#Building CNN Model
model = Sequential()

#First Layer
model.add(Conv2D(64, 3, activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Second Layer
model.add(Conv2D(32, 3, activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

#Final Output Layer
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
#Compiling Model
model.compile(optimizer='adam' , loss="categorical_crossentropy", metrics=["accuracy"])
#Training Model 
history = model.fit(x_train, y_train, batch_size = 64, epochs = 10, 
                    validation_data = (x_val, y_val), verbose = 2)
#Visualizing Model Accuracy

plt.title('Model Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.show()
predictions = model.predict(testing[:3])
#Printing prediction of model with labels of first 3 test images
print(np.argmax(predictions, axis = 1))
def prep_test_data(raw):
    x = raw[:,0:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, 28, 28, 1)
    out_x = out_x / 255
    return out_x

val_file = "/kaggle/input/digit-recognizer/test.csv"
val_data = np.loadtxt(val_file, skiprows=1, delimiter=',')
x_test = prep_test_data(val_data)
predictions = model.predict_classes(x_test)

indexes = [i for i in range(1,len(val_data)+1)]
output = pd.DataFrame({'ImageId': indexes,'Label': predictions})
output.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')
