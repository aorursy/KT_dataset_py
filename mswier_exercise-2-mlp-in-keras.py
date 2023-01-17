# to do OHE
from keras.utils import to_categorical
# the only python lib we really need
import numpy as np
# Keras tools
from keras.models import Sequential
from keras.layers import Dense, Flatten
# dataset
from sklearn import datasets
# plot / image show
from matplotlib import pyplot as plt
# to split data into trainign and test set
from sklearn.model_selection import train_test_split
lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, height, width = lfw_people.images.shape
print(n_samples, height, width)
#example image:
person_number = 6
plt.imshow( lfw_people.images[person_number] , cmap='gray')
plt.show()
print("target class: " +str(lfw_people.target[person_number]))
print("Name: " + str(lfw_people.target_names[lfw_people.target[person_number]]))
# do features
X = lfw_people.data
y = lfw_people.target

features_number = X.shape[1]
target_names = lfw_people.target_names
target_names
# scale X
X = X / 255.0
np.unique(y,return_counts=True)
# do OHE
y = to_categorical(y)
y[6]
# data set split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(X_train.shape)
print(X_test.shape)
# reshape 
X_train = X_train.reshape(772, height, width)
X_test = X_test.reshape(516, height, width)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
model = Sequential()

model.add(Flatten(input_shape=(50,37)))
# TODO : Add few Keras dense layers nad final layer

print(model.summary())
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics   = ['acc'] )
model.fit(X_train, y_train, batch_size = 16, epochs = 50, verbose = True)
print("Test score")
score = model.evaluate(X_test, y_test)
print(score)
print("Train set score")
score = model.evaluate(X_train, y_train)
print(score)
# using early stopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import callbacks
callbacks = [EarlyStopping(monitor='loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]
model = Sequential()

model.add(Flatten(input_shape=(50,37)))
# TODO : Copy your model here

print(model.summary())
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics   = ['acc'] )
model.fit(X_train, y_train, batch_size = 16, epochs = 100, verbose = True, callbacks=callbacks)
# LOAD MODEL
from keras.models import load_model 
model = load_model('best_model.h5')
print("Test score")
score = model.evaluate(X_test, y_test)
print(score)
print("Train set score")
score = model.evaluate(X_train, y_train)
print(score)
from keras.layers import Dropout
# add drop out
model = Sequential()

model.add(Flatten(input_shape=(50,37)))
# TODO: Redesign your network to add Dropout(0.1) layer one or two.
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics   = ['acc'] )
callbacks = [EarlyStopping(monitor='loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]
model.fit(X_train, y_train, batch_size = 16, epochs = 100, verbose = True, callbacks=callbacks)
# LOAD MODEL
from keras.models import load_model 
model = load_model('best_model.h5')
print("Test score")
score = model.evaluate(X_test, y_test)
print(score)
print("Train set score")
score = model.evaluate(X_train, y_train)
print(score)