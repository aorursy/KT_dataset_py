# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
import seaborn as sns
sns.countplot(y)
# Check IF some Feature variables are NaN
np.unique(np.isnan(X))[0]
# Check IF some Target Variables are NaN
np.unique(np.isnan(y))[0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 784, activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dropout(0.05))
# Adding the output layer
classifier.add(Dense(units = 10, activation = 'softmax'))
# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, 
                         validation_data = (X_test, y_test), 
                         batch_size = 28, 
                         epochs = 25)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model_acc = classifier.evaluate(X_test, y_test)
print(" Model Accuracy is : {0:.1f}%".format(model_acc[1]*100))
test_dataset = pd.read_csv('../input/test.csv')
test = test_dataset.iloc[:,:].values
# Prediction
test_pred = classifier.predict(test)

# Mark probability score > 0.5 as Predicted Label, axis = 1 means insert column-wise 
results = test_pred.argmax(axis=1)
for i in range(1,10):
    index = np.random.randint(1,28001)
    plt.subplot(3,3,i)
    plt.imshow(test[index].reshape(28,28))
    plt.title("Predicted Label : {}".format(results[index]))
plt.subplots_adjust(hspace = 1.2, wspace = 1.2)
plt.show()
    
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)
submission.head()