# Loading required packages

import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
# Load the data

train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.head()
train_data.shape
train_data.isnull().any().describe()
test_data.isnull().any().describe()
train_data['label'].value_counts()
plt.figure(figsize=(15,7))

sns.countplot(train_data['label'])

plt.title("Number of digit classes")
X = train_data.drop('label' , axis=1)

y = train_data['label']
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
X_train = X_train.values.reshape(-1, 28, 28, 1).astype('float32')

X_test = X_test.values.reshape(-1, 28, 28, 1).astype('float32') 
print(X_train.shape)

print(X_test.shape)
y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes = 10)
print(y_train.shape)

print(y_test.shape)
# Setting up the neural network model



NUM_CLASSES = 10

IMG_ROWS = 28

IMG_COLS = 28



model = Sequential()



model.add(Conv2D(32,(3,3),activation= 'relu',input_shape=(IMG_ROWS,IMG_COLS,1)))

model.add(Dropout(0.5))

model.add(Conv2D(32,(3,3),activation= 'relu'))

model.add(Dropout(0.5))

model.add(Conv2D(32,(3,3),activation= 'relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256,activation= 'relu'))

model.add(Dense(256,activation= 'relu'))

model.add(Dense(256,activation= 'relu'))

model.add(Dense(NUM_CLASSES,activation= 'softmax'))



model.summary()
# Compiling and fitting the model



model.compile(optimizer = 'adam', 

             loss = 'categorical_crossentropy',

             metrics = ['accuracy'])



model.fit(X_train, y_train, validation_split=0.2, epochs=10)
# Evaluating the model for out of sample performance



model.evaluate(X_test, y_test, batch_size=32)
# Reshaping the data for CNN



test_data_OH = test_data.values.reshape(-1, 28, 28, 1).astype('float32')

print(test_data_OH.shape)
# Prediction Dataset



predictions = model.predict(test_data_OH, batch_size = 32)

predictions = np.array([np.argmax(row) for row in predictions])

submission = pd.DataFrame({'ImageId' : np.arange(1,len(test_data)+1), 'Label' : predictions})
# Export to csv



submission.to_csv("cnn_mnist_predictions_12May2020.csv",index=False)