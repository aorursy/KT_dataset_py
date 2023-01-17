
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, math
from sklearn import neighbors, svm
import tensorflow as tf
# Input data files are available in the read-only "../input/" directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# https://www.kaggle.com/dansbecker/finding-your-files-in-kaggle-kernels

#Import training set and separate labels from data
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
y = train['label']
X = train.drop('label', axis='columns')

#Import test set
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
#This function formats predictions for kaggle submission
#predictions must be an numpy array
def write_submission(predictions, fname):
    if type(predictions) is np.ndarray:
        output = pd.DataFrame()
        output['ImageId'] = range(1,len(predictions) + 1)
        output['Label'] = predictions
        output.to_csv(os.path.join('/kaggle/working',fname), index_label=False, index=False)
    else:
        raise ValueError('predictions must be a numpy array. %s provided instead' % type(predictions))

#This function converts digit images to a different format
#input format - pandas DataFrame where each image is a 1d pixel vector
#output format - np.array where each image is a square image matrix
def unflatten(kaggle_df):
    ret = []
    kaggle_array = np.array(kaggle_df)
    dim = int(math.sqrt(kaggle_array.shape[1]))
    for i in kaggle_array:
        ret.append(i.reshape(dim, dim))
    return np.array(ret)
#Train a KNN classifier using sklearn
classifier = neighbors.KNeighborsClassifier(n_neighbors=10)
classifier.fit(X[:5000],y[:5000])
#Check classifier accuracy on a subset of test data.
#Running .score on all test data takes a long time to finish
classifier.score(X.iloc[500:1500],y.iloc[500:1500])
#Generate predictions for all rows in test set
#Future work: figure out why this takes significantly longer than classifier.fit()
predictions = classifier.predict(test)
write_submission(predictions, 'knn_predictions.csv')
#This is also pretty slow so only training on a subset of the data.
classifier = svm.SVC()
classifier.fit(X[:5000],y[:5000])
#Generate predictions for all rows in test set
predictions = classifier.predict(test)
write_submission(predictions, 'svc_predictions.csv')
#Normalize input for Neural Networks
#Reshape input data from a collection of flat pixel vectors to a collection of square images
X_nn = unflatten(X)
test_nn = unflatten(test)

#Map pixel values to interval [0,1]
X_nn = X_nn / 255.0
test_nn = test_nn / 255.0
# Define Traditional Neural Network model layers using keras Functional API
inputs = tf.keras.Input(shape=(28, 28, 1))
temp = tf.keras.layers.Flatten()(inputs)
temp = tf.keras.layers.Dense(128, activation="relu")(temp)
outputs = tf.keras.layers.Dense(10, activation="softmax")(temp)

# Instantiate model instance
nn_model = tf.keras.Model(inputs, outputs)

# Compile the model
nn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model
nn_model.fit(X_nn, y, batch_size=64, epochs=1)

#Generate predictions for all rows in test set
prediction_probs = nn_model.predict(test_nn)
predictions = np.array([np.argmax(p) for p in prediction_probs])
write_submission(predictions, 'nn_predictions.csv')
# Define Convolutional Neural Network model layers using keras Functional API
inputs = tf.keras.Input(shape=(28, 28, 1))
temp = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
temp = tf.keras.layers.MaxPooling2D(2,2)(temp)
temp = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(temp)
temp = tf.keras.layers.MaxPooling2D(2,2)(temp)
temp = tf.keras.layers.Flatten()(temp)
temp = tf.keras.layers.Dense(128, activation="relu")(temp)
outputs = tf.keras.layers.Dense(10, activation="softmax")(temp)

# Instantiate model instance
cnn_model = tf.keras.Model(inputs, outputs)

# Compile the model
cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model
cnn_model.fit(X_nn, y, batch_size=64, epochs=1)

#Generate predictions for all rows in test set
prediction_probs = cnn_model.predict(test_nn)
predictions = np.array([np.argmax(p) for p in prediction_probs])
write_submission(predictions, 'cnn_predictions.csv')