import numpy as np

import pandas as pd

import os

import keras

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization,Activation

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras import optimizers

from keras import regularizers

from sklearn.model_selection import train_test_split

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))
train_dir = '../input/sign-language/data/train/asl_alphabet_train'

test_dir = '../input/sign-language/data/val/asl_alphabet_test'
def load_unique():

    size_img = 64,64 

    images_for_plot = []

    labels_for_plot = []

    for folder in os.listdir(train_dir):

        for file in os.listdir(train_dir + '/' + folder):

            filepath = train_dir + '/' + folder + '/' + file

            image = cv2.imread(filepath)

            final_img = cv2.resize(image, size_img)

            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

            images_for_plot.append(final_img)

            labels_for_plot.append(folder)

            break

    return images_for_plot, labels_for_plot



images_for_plot, labels_for_plot = load_unique()

print("unique_labels = ", labels_for_plot)



fig = plt.figure(figsize = (15,15))

def plot_images(fig, image, label, row, col, index):

    fig.add_subplot(row, col, index)

    plt.axis('off')

    plt.imshow(image)

    plt.title(label)

    return

image_index = 0

row = 5

col = 6

for i in range(1,(row*col)):

    plot_images(fig, images_for_plot[image_index], labels_for_plot[image_index], row, col, i)

    image_index = image_index + 1

plt.show()
labels_dict = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,'K':20,'L':21,'M':22,

                   'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,'W':32,'X':33,'Y':34,

                   'Z':35,'space':36,'del':37,'nothing':38}

def load_data():

    """

    Loads data and preprocess. Returns train and test data along with labels.

    """

    i = 0

    images = []

    labels = []

    size = 64,64

    print("LOADING DATA FROM : ",end = "")

    

    for folder in os.listdir(train_dir):

        

            print(folder, end = ' | ')

            for image in os.listdir(train_dir + "/" + folder):

                temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)

                temp_img = cv2.resize(temp_img, size)

                images.append(temp_img)

                labels.append(labels_dict[folder])

    

    images = np.array(images)

    

    images = images.astype('float32')/255.0

    

    labels = keras.utils.to_categorical(labels)

    

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.05)

    

    print()

    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)

    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)

    

    

    return X_train, X_test, Y_train, Y_test

    

X_train, X_test, Y_train, Y_test = load_data()

def create_model():    

    model = Sequential()

    

    model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = (64,64,3)))

    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))

    model.add(Dense(39, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

    

    print("MODEL CREATED")

    model.summary()

    

    return model



def fit_model():

    model_hist = model.fit(X_train, Y_train, batch_size = 8, epochs = 5, validation_split = 0.2)

    return model_hist 
model = create_model()

curr_model_hist = fit_model()
plt.plot(curr_model_hist.history['accuracy'])

plt.plot(curr_model_hist.history['val_accuracy'])

plt.legend(['train', 'test'], loc='lower right')

plt.title('accuracy plot - train vs test')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.show()



plt.plot(curr_model_hist.history['loss'])

plt.plot(curr_model_hist.history['val_loss'])

plt.legend(['training loss', 'validation loss'], loc = 'upper right')

plt.title('loss plot - training vs vaidation')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
evaluate_metrics = model.evaluate(X_test, Y_test)

print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
def load_test_data():

    images = []

    names = []

    size = 64,64

    for image in os.listdir(test_dir):

        temp = cv2.imread(test_dir + '/' + image)

        temp = cv2.resize(temp, size)

        images.append(temp)

        names.append(image)

    images = np.array(images)

    images = images.astype('float32')/255.0

    return images, names



test_images, test_img_names = load_test_data()
predictions = [model.predict_classes(image.reshape(1,64,64,3))[0] for image in test_images]
def get_labels_for_plot(predictions):

    predictions_labels = []

    for i in range(len(predictions)):

        for ins in labels_dict:

            if predictions[i] == labels_dict[ins]:

                predictions_labels.append(ins)

                break

    return predictions_labels



predictions_labels_plot = get_labels_for_plot(predictions)
predfigure = plt.figure(figsize = (13,13))

def plot_image_1(fig, image, label, prediction, predictions_label, row, col, index):

    fig.add_subplot(row, col, index)

    plt.axis('off')

    plt.imshow(image)

    title = "prediction : [" + str(predictions_label) + "] "+ "\n" + label

    plt.title(title)

    return



image_index = 0

row = 5

col = 6

for i in range(1,(row*col-1)):

    plot_image_1(predfigure, test_images[image_index], test_img_names[image_index], predictions[image_index], predictions_labels_plot[image_index], row, col, i)

    image_index = image_index + 1

plt.show()
model.save('model.h5')