#Loading libraries and methods:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from tqdm import tqdm
#Loading the dataset:
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.head()
#Dataset dimensions:
data.shape
#The pred variable stores the predictors (petal.length, ...).
#The cla variable stores the classes.

#Converting to the numpy format:
pred = data.iloc[:, 0:4].values

#Converting to the numpy format:
cla = data.iloc[:, 4].values 

#Transforming to numeric features:
#labelencoder = LabelEncoder()
#cla = labelencoder.fit_transform(cla)

cla_dummy = np_utils.to_categorical(cla, 3)

#Iris setosa      1 0 0
#Iris virginica   0 1 0
#Iris versicolor  0 0 1
#Splitting the dataset into training and test datasets:

pred_train, pred_test, cla_train, cla_test = train_test_split(pred, cla_dummy, test_size = 0.25)
#Training dataset: 75% / Test dataset: 25%

print(pred_train.shape)
print(pred_test.shape)
print(cla_train.shape)
print(cla_test.shape)
#Building the neural network with 2 hidden layers:

classifier = Sequential()
classifier.add(Dense(units=4, activation='relu', input_dim = 4))
classifier.add(Dense(units=4, activation='relu'))
classifier.add(Dense(units=3, activation='softmax'))

#Compiling the neural network:
classifier.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
#Neural network architecture:

from IPython.display import Image
Image(filename='/kaggle/input/images/meuralnet.png') 
#Training the neural network:

tqdm(classifier.fit(pred_train, cla_train, batch_size = 10, epochs = 1000))
#applying the trained model to the test set:

results = classifier.evaluate(pred_test, cla_test)
#Predictions:

prediction = classifier.predict(pred_test)
prediction = (prediction > 0.5)
#Confusion matrix:

cla_test2 = [np.argmax(t) for t in cla_test]
prediction2 = [np.argmax(t) for t in prediction]

matrix = confusion_matrix(prediction2, cla_test2)
matrix
