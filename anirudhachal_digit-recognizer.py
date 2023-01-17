# Importing header files
import os
import pandas as pd
import numpy as np
# Setting path
train_data_path = "../input/digit-recognizer/train.csv"
test_data_path = "../input/digit-recognizer/test.csv"
# Load data frames
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
train_df
test_df
train_df.info(verbose = True)
test_df.info(verbose = True)
print("Max value in train_df :", max(train_df.max()))
print("Min value in train_df :", min(train_df.min()))
print("Max value in test_df :", max(test_df.max()))
print("Min value in test_df :", min(test_df.min()))
train_X = train_df.loc[:, 'pixel0':].to_numpy()
train_y = train_df.loc[:, 'label'].to_numpy().reshape(-1,1).ravel()
test_X = test_df.to_numpy()
train_X = np.divide(train_X, 255)
test_X = np.divide(test_X, 255)
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
# Function to get submission_file
def get_submission_file(model, filename):
    predictions = model.predict(test_X)
    df_submission = pd.DataFrame({'ImageId' : test_df.index + 1, 'Label' : predictions})
    # submission file
    submission_data_path = os.path.join(os.path.pardir, 'Data', 'Predictions')
    submission_file_path = os.path.join(submission_data_path, filename)
    # write to the file
    df_submission.to_csv(submission_file_path, index = False)
# Function to test model
from sklearn.model_selection import train_test_split

def test_model(model, trials):
    total_score = 0
    for trial in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size = 0.2)
        model.fit(X_train,  y_train)
        total_score += model.score(X_test, y_test)
    print(f"Average Accuracy of the model : {round(total_score / trials, 3)}") 
from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(strategy = "most_frequent")

test_model(dummy_model, 10)
from sklearn.linear_model import LogisticRegression

lr_model_1 = LogisticRegression(max_iter = 1000)

test_model(lr_model_1, 1)
from sklearn.neighbors import KNeighborsClassifier

KCN_1 = KNeighborsClassifier()

test_model(KCN_1, 1)
from sklearn.ensemble import RandomForestClassifier

RFC_1 = RandomForestClassifier()

test_model(RFC_1, 1)
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB();

test_model(GNB, 10)
from sklearn.neural_network import MLPClassifier

clf_1 = MLPClassifier()

test_model(clf_1, 1)
from sklearn.neural_network import MLPClassifier

clf_2 = MLPClassifier(hidden_layer_sizes = (100, 100))

test_model(clf_2, 1)
from sklearn.neural_network import MLPClassifier

clf_3 = MLPClassifier(hidden_layer_sizes = (62, 62))

test_model(clf_3, 1)
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

test_model(DTC, 10)
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

test_model(RFC, 1)
import tensorflow as tf
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size = 0.2)
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)
X = test_X.reshape(-1, 28, 28)
# Function to get submission_file
def get_submission_file_TF(model, filename):
    y = model.predict(X)
    predictions = []
    for prediction in y:
        predictions.append(np.argmax(prediction))
    df_submission = pd.DataFrame({'ImageId' : test_df.index + 1, 'Label' : predictions})
    # submission file
    submission_data_path = os.path.join(os.path.pardir, 'Data', 'Predictions')
    submission_file_path = os.path.join(submission_data_path, filename)
    # write to the file
    df_submission.to_csv(submission_file_path, index = False)
def get_submission_file_TF_cnn(model, filename):
    y = model.predict(X.reshape(-1, 28, 28, 1))
    predictions = []
    for prediction in y:
        predictions.append(np.argmax(prediction))
    df_submission = pd.DataFrame({'ImageId' : test_df.index + 1, 'Label' : predictions})
    # submission file
    submission_data_path = os.path.join(os.path.pardir, 'Data', 'Predictions')
    submission_file_path = os.path.join(submission_data_path, filename)
    # write to the file
    df_submission.to_csv(submission_file_path, index = False)
def get_predictions(model, X):
    y = model.predict(X)
    predictions = []
    for prediction in y:
        predictions.append(np.argmax(prediction))
    return predictions
def evaluate_model(model, X, y):
    val_loss, val_acc = model.evaluate(X, y)
    print(val_loss, val_acc)
def print_image(image):
    plt.imshow(image, cmap = 'gray')
    plt.show()
print_image(X_train[0])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation = tf.nn.sigmoid))
model.add(Dense(128, activation = tf.nn.sigmoid))
model.add(Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 20)
evaluate_model(model, X_test, y_test)
predictions = get_predictions(model, X_test)
def get_wrong_predictions(predictons):
    wrong_predictions = []
    for i in range(len(predictions)):
        if(y_test[i] != predictions[i]):
             wrong_predictions.append(i)
    return wrong_predictions
wrong_predictions = get_wrong_predictions(predictions)
import random
def get_a_wrong_prediction(wrong_predictions):
    n = random.randint(0, len(wrong_predictions))
    n = wrong_predictions[n]
    print_image(X_test[n])
    print("Prediction :", predictions[n])
    print("Expected :", y_test[n])
get_a_wrong_prediction(wrong_predictions)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

model = Sequential()

# Layer 1
model.add(Conv2D(128, (3, 3), input_shape = (28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Layer 2
model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
       
# Layer 3
model.add(Flatten())
model.add(Dense(128))
          
# Output Layer
model.add(Dense(10))
model.add(Activation('sigmoid'))
          
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
          
model.fit(X_train.reshape(-1, 28, 28, 1), y_train, batch_size = 32, epochs = 10)
evaluate_model(model, X_test.reshape(-1, 28, 28, 1), y_test)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2)

datagen.fit(X_train.reshape(-1, 28, 28, 1))

model.fit(datagen.flow(X_train.reshape(-1, 28, 28, 1), y_train, batch_size = 32), batch_size = 32, steps_per_epoch = (len(X_train) / 32), epochs = 10)
evaluate_model(model, X_test.reshape(-1, 28, 28, 1), y_test)