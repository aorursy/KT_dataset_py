# Importing "hot" library to handle files and datasets with Python.
import pandas as pd

# Asking to Jupyter to show images and graphics hight here (not in other places or windows).
%matplotlib inline
# Loading the dataset we want to take a look.
dataset = pd.read_csv("../input/digit-recognizer/train.csv")
# Looking at the shape of the dataset: 42.000 rows (registers) x 785 columns (attributes).
print(dataset.shape)
# Verifing how first registers look like.
dataset.head()
# Separating input values from labels.
y_train = dataset['label']
x_train = dataset.drop('label', axis=1)

# Releasing memory space (a good practice).
del dataset
# Starting by input values analysis.
x_train.describe()
# Hey! It looks that some attributes (features) have "fixed" values for all their registers.
# This kind of attributes doesn't contribute with relevant information. They can't tell us anything (they never change!).
x_train['pixel0'].plot()
# We can remove those fixed features from our input dataset.
# But we have to find a way to remember them later...
dropped_columns = []
for column in x_train.columns:
    if x_train[column].max() == x_train[column].min():
        dropped_columns.append(column)   
x_train.drop(dropped_columns, axis = 1, inplace = True) 
print('Dropped columns:', len(dropped_columns))
print('New shape of training dataset:', x_train.shape)
# Confirming we don't have null values as input.
# Obs. We have to know what to do with them in case they exist.
for column in x_train.columns:
    if x_train[column].isnull().any():
        print('Null value detected in the feature:', column)
# Now we can normalize the input values. Feature by feature.
# Obs.: Normally, low values are easier to learn. That is the reason we are doing the normalization.
min_train = {}
max_train = {}
for column in x_train.columns:
    min_train[column] = x_train[column].min()
    max_train[column] = x_train[column].max()
    x_train[column] = (x_train[column] - x_train[column].min()) / (x_train[column].max() - x_train[column].min()) # saving amplitudes
x_train = x_train.values
# And what about the labels?
# Let us take a look at the labels...
y_train.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=False)
# We have 10 distinct labels (0-9). It means we can classify the inputs instances in ten distinct classes.
# They are reasonably well distributed.
# Lets use a function from Keras to encode those labels into a map of 'ones'.
# That encoding is named 'One Hot Encoding'.
# We are doing that encoding because it is more easy to deal with map of ones as output when we intend to use a neural network as a classifier.
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
print(y_train)
# Now we can buid a ML model with 708 inputs and 10 outputs.
# Let us use Keras library for that.
from keras.models import Sequential
FEATURES = 708
LABELS = 10
# Let's start using a Logistic Regression model as our machine learning solution.
from keras.layers.core import Dense, Activation
model_LR = Sequential()
model_LR.add(Dense(LABELS, input_shape = (FEATURES,)))
model_LR.add(Activation('softmax'))
model_LR.summary()
# We will use:
#   Stochastic Gradient Descent to adjust coefficients,
#   Cross Entropy to get the loss and
#   Accuracy as our model's performance metrics.
from keras.optimizers import SGD
model_LR.compile(loss = 'categorical_crossentropy', optimizer = SGD(), metrics = ['accuracy'])
# Defining training parameters
EPOCHS = 50
BATCH_SIZE = 100
VALIDATION_SIZE = 0.1
# Starting the model's training
training_history = model_LR.fit(x_train,
                                y_train,
                                batch_size = BATCH_SIZE,
                                epochs = EPOCHS,
                                verbose = 1,
                                validation_split = VALIDATION_SIZE)
# Verifing the training results
from matplotlib import pyplot as plt

fig, ax = plt.subplots(2,1)
ax[0].plot(training_history.history['loss'], color='b', label="Training loss")
ax[0].plot(training_history.history['val_loss'], color='r', label="validation loss",axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(training_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(training_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# I think we can do better...
# Let's try to use a Neural Network with 2 layers as a solution model. Each layer'll have 128 neurons.
HIDDEN = 128
model_NN = Sequential()
model_NN.add(Dense(HIDDEN, input_shape = (FEATURES,)))
model_NN.add(Activation('relu'))
model_NN.add(Dense(HIDDEN))
model_NN.add(Activation('relu'))
model_NN.add(Dense(LABELS))
model_NN.add(Activation('softmax'))
model_NN.summary()

# In this case we'll use Adam algorithm to adjust the neurons' connections weights
from keras.optimizers import Adam
model_NN.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
# Starting the model's training
training_history = model_NN.fit(x_train,
                                y_train,
                                batch_size = BATCH_SIZE,
                                epochs = EPOCHS,
                                verbose = 1,
                                validation_split = VALIDATION_SIZE)
# Verifing the training results
fig, ax = plt.subplots(2,1)
ax[0].plot(training_history.history['loss'], color='b', label="Training loss")
ax[0].plot(training_history.history['val_loss'], color='r', label="validation loss",axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(training_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(training_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Comparing the models, the last one presents better results.
del model_LR
# And if we try to use the dropout's technique as way to prevent overfiting and to give some more generalization power to our model?
from keras.layers.core import Dropout 
DROPOUT = 0.25

model_NNv2 = Sequential()
model_NNv2.add(Dense(HIDDEN, input_shape = (FEATURES,)))
model_NNv2.add(Activation('relu'))
model_NNv2.add(Dropout(DROPOUT))
model_NNv2.add(Dense(HIDDEN))
model_NNv2.add(Activation('relu'))
model_NNv2.add(Dropout(DROPOUT))
model_NNv2.add(Dense(LABELS))
model_NNv2.add(Activation('softmax'))
model_NNv2.summary()
model_NNv2.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
# Starting the model's training.
training_history = model_NNv2.fit(x_train,
                                   y_train,
                                   batch_size = BATCH_SIZE,
                                   epochs = EPOCHS,
                                   verbose = 1,
                                   validation_split = VALIDATION_SIZE)
# Verifing the training results
fig, ax = plt.subplots(2,1)
ax[0].plot(training_history.history['loss'], color='b', label="Training loss")
ax[0].plot(training_history.history['val_loss'], color='r', label="validation loss",axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(training_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(training_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# The dropout only helps to keep the curves more stable
del model_NN
# It looks like the dropout didn't help much. Let's see what happens if we put some extra neurons to work?
HIDDEN_L1 = 256
HIDDEN_L2 = 128
model_NNv3 = Sequential()
model_NNv3.add(Dense(HIDDEN_L1, input_shape = (FEATURES,)))
model_NNv3.add(Activation('relu'))
model_NNv3.add(Dropout(DROPOUT))
model_NNv3.add(Dense(HIDDEN_L2))
model_NNv3.add(Activation('relu'))
model_NNv3.add(Dropout(DROPOUT))
model_NNv3.add(Dense(LABELS))
model_NNv3.add(Activation('softmax'))
model_NNv3.summary()
model_NNv3.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
# Starting the model's training.
training_history = model_NNv3.fit(x_train,
                                   y_train,
                                   batch_size = BATCH_SIZE,
                                   epochs = EPOCHS,
                                   verbose = 1,
                                   validation_split = VALIDATION_SIZE)
# Verifing the training results
fig, ax = plt.subplots(2,1)
ax[0].plot(training_history.history['loss'], color='b', label="Training loss")
ax[0].plot(training_history.history['val_loss'], color='r', label="validation loss",axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(training_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(training_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Comparing the models, the last one presents a little better results.
del model_NNv2
# Let's make the last attempt.
model_NNv4 = Sequential()
model_NNv4.add(Dense(HIDDEN_L1, input_shape = (FEATURES,)))
model_NNv4.add(Activation('relu'))
model_NNv4.add(Dropout(DROPOUT))
model_NNv4.add(Dense(HIDDEN_L2))
model_NNv4.add(Activation('relu'))
model_NNv4.add(Dropout(DROPOUT))
model_NNv4.add(Dense(LABELS))
model_NNv4.add(Activation('softmax'))
model_NNv4.summary()

# In this case we'll use RMSprop algorithm to adjust the neurons' connections weights
from keras.optimizers import RMSprop
model_NNv4.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(), metrics = ['accuracy'])
# Starting the model's training.
training_history = model_NNv4.fit(x_train,
                                   y_train,
                                   batch_size = BATCH_SIZE,
                                   epochs = EPOCHS,
                                   verbose = 1,
                                   validation_split = VALIDATION_SIZE)
# Verifing the training results
fig, ax = plt.subplots(2,1)
ax[0].plot(training_history.history['loss'], color='b', label="Training loss")
ax[0].plot(training_history.history['val_loss'], color='r', label="validation loss",axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(training_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(training_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# The new optimizer didn't help much
del model_NNv4
# Now, we have to apply the Kaggle's test data to our better NN model in order to obtain the predictions. 
x_test = pd.read_csv("../input/digit-recognizer/test.csv")
print('Shape of testing dataset:', x_test.shape)
# Do you remeber we had cut/dropped/ignored some features from training data before we started the construction of our model? Now, we have to do the same thing with the test data.
dropped_test_columns = x_test[dropped_columns] 
print('Shape of dropped test dataset:', dropped_test_columns.shape)
# Wait a minute! We'll lost some test information! It's not a good notice!(?)
for column in dropped_test_columns:
    if not dropped_test_columns[column].min() == dropped_test_columns[column].max():
        print(column, dropped_test_columns[column].min(), dropped_test_columns[column].max())
# Well, let's continue our exercise and see if we really did a bad/good thing...
x_test.drop(dropped_columns, axis = 1, inplace = True)
print('Dropped columns:', len(dropped_columns))
print('New shape of testing dataset:', x_test.shape)
# Checking for null values.
for column in x_test.columns:
    if x_test[column].isnull().any():
        print('Null value detected in the feature:', column)
# Normalizing the test data like we did with training data. We have to pay attention here because we have to use the same amplitudes/limites we had used for training phase.
for column in x_test.columns:
    x_test[column] = (x_test[column] - min_train[column]) / (max_train[column] - min_train[column])
# Getting the predictions
x_test = x_test.values
results = model_NNv3.predict(x_test)
# Putting the predictions in terms of 'ones' (one hot enconding).
from numpy import argmax
results = argmax(results, axis = 1)
results = pd.Series(results, name = "Label")
# Saving the predicitions for submission.
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"),results], axis = 1)
submission.to_csv("submission.csv", index = False)
# Releasing memory space.
# del model_NNv3,
#     x_train,
#     y_train,
#     min_train,
#     max_train,
#     training_history,
#     dropped_columns,
#     dropped_test_columns,
#     x_test,
#     results
# Conclusion:
# Despite we had cut out 76/784 = 9,7% of all training and test data (features) and had ignored the nature of those data (pictures/images) we got pretty good results (Kaggle's score 0.98). Sometimes, we can ignore some input data offered to us without breaking the quality limits of the results we need. Data scientist must know how to choose which data are actually relevant.