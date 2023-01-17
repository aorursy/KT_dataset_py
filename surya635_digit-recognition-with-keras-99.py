#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
%matplotlib inline
#load data
digit = pd.read_csv('../input/train.csv')
digit.head()
#shape
digit.shape
digit.isnull().any().describe()
#let's seprate the target variable
y = digit['label'].values
del digit['label']
#count plot of target variable
sb.countplot(y)
plt.show()
#Normalization
digit = digit/255.0
#Reshape the data
X = digit.values.reshape(len(digit), 28, 28, 1)
#Converting target to categorical features
from keras.utils.np_utils import to_categorical
y = to_categorical(y)
y  #categorical formate
#split the data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#train model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#summary of model
model.summary()
#fitting the model
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'], loc='lower right')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
#Predicting test data
y_pred = model.predict_classes(X_test)
#Arrange
y_test_cls = np.argmax(y_test, axis=1)
#Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred, y_test_cls)
plt.figure(figsize=(8, 5))
sb.set(font_scale=1.2)
sb.heatmap(cm, annot=True, fmt='g')
plt.show()
# Evaluate data
score = model.evaluate(X_test, y_test)
print('Accuracy : ', score[1]*100)