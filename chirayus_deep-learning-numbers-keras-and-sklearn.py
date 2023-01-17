# Data analysis

import pandas as pd

import numpy as np

from scipy import stats, integrate



# Scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder



# Keras

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

from keras.utils import np_utils



# Data visualization

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# List of files available in the input directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# The training set is quite large (75 MB), we read only the first 10 records for now

train = pd.read_csv('../input/train.csv', nrows=10)
train
# Display the first 'n' images of a data set

f = plt.figure(figsize=(10,4))

# Loop over the images of the training set

for i in range(10):

    ax = plt.subplot(2, 5, i+1)

    ax.axis('off')

    image = train.iloc[i, 1:].values.reshape(28, 28)

    ax.set_title(train['label'].iloc[i])

    imgplot = plt.imshow(image, cmap='binary')



plt.show()
# Load images in memory

train = pd.read_csv('../input/train.csv', nrows=10000)

labels = train.iloc[:,0].values.astype('int32')

X_train = (train.iloc[:, 1:].values).astype('float32') # Pixel intensities of the images

X_test = (pd.read_csv('../input/test.csv').values).astype('float32')



y_train = np_utils.to_categorical(labels) # Labels as categories



# Convert images to black and white

#X_train[X_train > 0] = 1

scale = np.max(X_train)

X_train /= scale

X_test /= scale



mean = np.std(X_train)

X_train -= mean

X_test -= mean



# Split data for training and validation.

X_train, X_val, y_train, y_val = train_test_split(X_train, 

                                                  y_train, 

                                                  test_size=0.2, 

                                                  random_state=1)



print('X_train shape:', X_train.shape, '| y_train shape:', y_train.shape)

print('X_val shape:', X_val.shape, '| y_val shape:', y_val.shape)
# Instantiate the MLP classifier

clf = MLPClassifier(solver='lbfgs', # Limited-memory BFGS optimizer

                    activation='relu', # Rectified linear unit as activation function, returns f(x) = max(0, x)

                    hidden_layer_sizes=(500, ) # One hidden layer of 500 nodes 

                   )



# Fit the model on the training set

clf.fit(X_train, y_train)
# Run prediction on the validation set 

y_pred = clf.predict(X_val)



# Print results

print('Classification report:\n\n', classification_report(y_val, y_pred), '\n')
# Insert our prediction in a DataFrame with the ImageId

y_pred = y_pred.argmax(1)

predictions = pd.DataFrame({'ImageId': list(range(1, y_pred.size+1)), 

                            'Label': y_pred.astype(int)}

                     )

print('Completed prediction of {} images'.format(predictions.shape[0]))
# Create figure

fig = plt.figure(figsize=(9,6))



# loop over the first 10 images of the test set

for i in range(25):

    ax = plt.subplot(5, 5, i+1)

    ax.set_title('Prediction: {}'.format(predictions.iloc[i].Label))

    ax.axis('off')

    image = X_val[i].reshape(28, 28)

    imgplot = plt.imshow(image, cmap='binary')



plt.show()
input_dim = X_train.shape[1]

nb_classes = y_train.shape[1]



# Here's a Deep Dumb MLP (DDMLP)

model = Sequential()

model.add(Dense(128, input_dim=input_dim))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



print("Training...")

model.fit(X_train, y_train, 

          nb_epoch=10, 

          batch_size=16, 

          validation_split=0.2, 

#          show_accuracy=True, 

          verbose=2)

print('------------------------------\n')

print("Training complete.")
print("Generating test predictions...")

y_pred = model.predict_classes(X_test, verbose=0)

#print("Complete.")



predictions = pd.DataFrame({'ImageId': list(range(1, y_pred.size+1)), 

                            'Label': y_pred.astype(int)}

                     )

print('Completed prediction of {} images'.format(predictions.shape[0]))
# Create figure

fig = plt.figure(figsize=(9,6))



# Loop over some images of the test set

for i in range(40):

    ax = plt.subplot(8, 5, i+1)

    ax.set_title('Prediction: {}'.format(predictions.iloc[i].Label))

    ax.axis('off')

    image = X_test[i].reshape(28, 28)

    imgplot = plt.imshow(image, cmap='binary')



plt.show()