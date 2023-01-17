import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import numpy as np
import cv2
import pandas as pd 
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import h5py
from keras.models import load_model
import csv

print(tf.__version__)
image_size = 28
PATH = '../input/Kannada-MNIST'

row_length = 28
column_length = 28

training_size = 0.85
validation_size = 0.10
test_size = 0.05
training_end_index_ratio = training_size
validation_end_index_ratio = training_size + validation_size

df_train = pd.read_csv(PATH + "/" + "train.csv")
df_dig = pd.read_csv(PATH + "/" + "Dig-MNIST.csv")

df_train = df_train.sample(frac=1).reset_index(drop=True)
df_dig = df_dig.sample(frac=1).reset_index(drop=True)

df_train_train, df_train_validation, df_train_test = np.split(df_train, [int(training_end_index_ratio * len(df_train)), int(validation_end_index_ratio * len(df_train))])
df_dig_train, df_dig_validation, df_dig_test = np.split(df_dig, [int(training_end_index_ratio * len(df_dig)), int(validation_end_index_ratio * len(df_dig))])
def get_output_vector(category):
  output_labels = np.zeros(10)
  output_labels[int(category)] = 1
  return np.array(output_labels, dtype=np.float32)

def get_formatted_labels(data_list):
    labels = []
    for i in range(0,len(data_list)):
        labels.append(get_output_vector(data_list[i]))
    return np.array(labels)

def convert_two_dimentional(single_list):
  return single_list.reshape((image_size, image_size))

def convert_grey2rgb(two_list):
  return cv2.merge([two_list, two_list, two_list])

def get_image(row):
  return convert_grey2rgb(convert_two_dimentional(np.array(row, dtype=np.float32)))

def format_list_data(df_data_1, df_data_2):
    data_1 = np.array(df_data_1.drop(columns="label"))
    data_2 = np.array(df_data_2.drop(columns="label"))
    list_data = []
    for i in range(0,len(data_1)):
        list_data.append(get_image(data_1[i]))
    for i in range(0,len(data_2)):
        list_data.append(get_image(data_2[i]))
    return np.array(list_data, dtype=np.float32) / 255
def get_train_data():
    return format_list_data(df_train_train, df_dig_train), get_formatted_labels(np.append(df_train_train["label"], df_dig_train["label"]))

def get_validation_data():
    return format_list_data(df_train_validation, df_dig_validation), get_formatted_labels(np.append(df_train_validation["label"], df_dig_validation["label"]))

def get_test_data():
    return format_list_data(df_train_test, df_dig_test), get_formatted_labels(np.append(df_train_test["label"], df_dig_test["label"]))

def format_submission_data(df_data):
    data = np.array(df_data.drop(columns="id"))
    list_data = []
    for i in range(0,len(data)):
        list_data.append(get_image(data[i]))
    return np.array(list_data, dtype=np.float32) / 255

def get_submission_test_data():
    df_test = pd.read_csv(PATH + "/" + "test.csv")
    return format_submission_data(df_test)
learning_rate = 0.001
epochs = 20
early_stop_constant = 7
batch_size_32 = 32
batch_size_64 = 64
batch_size_128 = 128
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.categorical_crossentropy
metrics = ['accuracy']
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   zoom_range=0.20,
                                   horizontal_flip=False)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   zoom_range=0.20,
                                   horizontal_flip=False)
REGULARIZATION_CONSTANT = 0
DROPOUT_CONSTANT = 0.2

input_layer = tf.keras.Input(shape=(image_size, image_size, 3), name="Input_Layer")

x = tf.keras.layers.Conv2D(filters=64, 
                           kernel_size=(3,3), 
                           padding="same", 
                           activation=tf.keras.activations.relu)(input_layer)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=64, 
                           kernel_size=(3,3), 
                           padding="same",
                           activation=tf.keras.activations.relu)(input_layer)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=64, 
                           kernel_size=(3,3), 
                           padding="same",
                           activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(DROPOUT_CONSTANT)(x)

x = tf.keras.layers.Conv2D(filters=128, 
                           kernel_size=(3,3), 
                           padding="same", 
                           activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=128, 
                           kernel_size=(3,3), 
                           padding="same", 
                           activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=128, 
                           kernel_size=(3,3), 
                           padding="same", 
                           activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(DROPOUT_CONSTANT)(x)

x = tf.keras.layers.Conv2D(filters=256,
                           kernel_size=(5,5), 
                           padding="same", 
                           activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(DROPOUT_CONSTANT)(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(units=1024, 
                          activation=tf.keras.activations.relu, 
                          name="Hidden_Layer_1")(x)
x = tf.keras.layers.Dropout(DROPOUT_CONSTANT)(x)
x = tf.keras.layers.Dense(units=512, 
                          activation=tf.keras.activations.relu, 
                          name="Hidden_Layer_2")(x)
x = tf.keras.layers.Dropout(DROPOUT_CONSTANT)(x)
output_layer = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(x)

def get_model():
    functional_model = tf.keras.Model(input_layer, output_layer)
    functional_model.summary()
    return functional_model

def exponential_lr_decay(epoch, initial_learningrate = 0.01):
    k = 0.001
    max_epoch_default_learningrate = 1
    if(epoch <= max_epoch_default_learningrate):
      return initial_learningrate
    else:
      return initial_learningrate * np.exp(-1 * k * epoch)

checkpoint = tf.keras.callbacks.ModelCheckpoint('my_model.h5', monitor='accuracy', save_best_only=True)

lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", 
                                            patience=3,
                                            factor=0.5,
                                            min_lr=0.000001)

class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, test_data, test_label):
        super(CustomCallback,self).__init__()
        self.test_data = test_data
        self.test_label = test_label
    
    def on_epoch_end(self, epoch, logs=None):
        self.find_test_accuracy(self.test_data, self.test_label)

    def is_equal(self, val1, val2):
        return val1 == val2

    def get_correct_count(self, y, yhat, current_count):
        if self.is_equal(y, yhat):
            return current_count + 1
        else:
            return current_count

    def find_test_accuracy(self, test_data, test_labels):
        yhat = np.argmax(self.model.predict(test_data), axis=1)
        y = np.argmax(test_labels, axis=1)
        correct_count = 0
        assert len(yhat) == len(y)
        for i in range(0,len(y)):
            correct_count = self.get_correct_count(y[i], yhat[i], correct_count)
        print("Test Data Accuracy: " + str(correct_count * 100 / len(y)) + "%") 

        
callbacks = [lr, checkpoint]
functional_model = get_model()
functional_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
train_data, train_labels = get_train_data()
validation_data, validation_labels = get_validation_data()
history = functional_model.fit_generator(train_datagen.flow(x=train_data, y=train_labels, batch_size=128), 
                               epochs=30, 
                               callbacks=callbacks,
                               validation_data=(validation_data, validation_labels))
def evaluate_test_data():
    test_data, test_labels = get_test_data()
    y = np.argmax(test_labels,axis=1)
    yhat =  np.argmax(functional_model.predict(test_data),axis=1)
    correct=0
    for i in range(0,len(y)):
        if y[i] == yhat[i]:
            correct = correct + 1
            
    print("Test Accuracy: " + str(correct * 100/ len(y)))
    
evaluate_test_data()
test_data = get_submission_test_data()
yhat =  np.argmax(functional_model.predict(test_data),axis=1)
df_submission = pd.read_csv(PATH + "/" + "sample_submission.csv")
df_submission["label"] = pd.Series(yhat)
df_submission.to_csv("submission.csv", index=False)
df_submission.head()
functional_model.save('my_model.h5')