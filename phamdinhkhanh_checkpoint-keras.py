from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

%matplotlib inline
seed = 7
np.random.seed(seed)

# load data
dataset = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter = ',')

X = dataset[:, 0:8]
y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = np.int(X.shape[0]/3), random_state = 123, stratify = y)

# create model
model = Sequential()
model.add(Dense(12, input_dim = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# summary model
model.summary()

# input shape
print('X train shape: ', X_train.shape)
print('y train shape: ', y_train.shape)

print('X test shape: ', X_test.shape)
print('y test shape: ', y_test.shape)
# compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# create checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = 'max')

callbacks_list = [checkpoint]
# fit the model
model.fit(X_train, y_train, 
          validation_split = 0.33, 
          epochs = 150, 
          batch_size = 10,
          callbacks = callbacks_list,
          verbose = 0)
# compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# create checkpoint
filepath="weights-best-file.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = 'max')

callbacks_list = [checkpoint]

# fit the model
model.fit(X_train, y_train, 
          validation_split = 0.33, 
          epochs = 150, 
          batch_size = 10,
          callbacks = callbacks_list,
          verbose = 0)
# create model
model1 = Sequential()
model1.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model1.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model1.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# load weights
model1.load_weights("weights-best-file.hdf5", by_name = True)
# Compile model (required to make predictions)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")

# estimate accuracy on whole dataset using loaded weights
scores = model1.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))
# create model
model2 = Sequential()
model2.add(Dense(24, input_dim=8, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(12, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(6, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# load weights
model2.load_weights("weights-best-file.hdf5", by_name = True)
# Compile model (required to make predictions)
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")

# estimate accuracy on whole dataset using loaded weights
scores = model2.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))