import numpy as np

import pandas as pd

import os

import keras

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras import regularizers

from sklearn.model_selection import train_test_split

import cv2

import matplotlib.pyplot as plt

print(os.listdir("../input"))
train_dir = '../input/asl_alphabet_train/asl_alphabet_train'

test_dir = '../input/asl_alphabet_test/asl_alphabet_test'
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
labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,

                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,

                   'Z':25,'space':26,'del':27,'nothing':28}



def load_data():

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

            if folder == 'A':

                labels.append(labels_dict['A'])

            elif folder == 'B':

                labels.append(labels_dict['B'])

            elif folder == 'C':

                labels.append(labels_dict['C'])

            elif folder == 'D':

                labels.append(labels_dict['D'])

            elif folder == 'E':

                labels.append(labels_dict['E'])

            elif folder == 'F':

                labels.append(labels_dict['F'])

            elif folder == 'G':

                labels.append(labels_dict['G'])

            elif folder == 'H':

                labels.append(labels_dict['H'])

            elif folder == 'I':

                labels.append(labels_dict['I'])

            elif folder == 'J':

                labels.append(labels_dict['J'])

            elif folder == 'K':

                labels.append(labels_dict['K'])

            elif folder == 'L':

                labels.append(labels_dict['L'])

            elif folder == 'M':

                labels.append(labels_dict['M'])

            elif folder == 'N':

                labels.append(labels_dict['N'])

            elif folder == 'O':

                labels.append(labels_dict['O'])

            elif folder == 'P':

                labels.append(labels_dict['P'])

            elif folder == 'Q':

                labels.append(labels_dict['Q'])

            elif folder == 'R':

                labels.append(labels_dict['R'])

            elif folder == 'S':

                labels.append(labels_dict['S'])

            elif folder == 'T':

                labels.append(labels_dict['T'])

            elif folder == 'U':

                labels.append(labels_dict['U'])

            elif folder == 'V':

                labels.append(labels_dict['V'])

            elif folder == 'W':

                labels.append(labels_dict['W'])

            elif folder == 'X':

                labels.append(labels_dict['X'])

            elif folder == 'Y':

                labels.append(labels_dict['Y'])

            elif folder == 'Z':

                labels.append(labels_dict['Z'])

            elif folder == 'space':

                labels.append(labels_dict['space'])

            elif folder == 'del':

                labels.append(labels_dict['del'])

            elif folder == 'nothing':

                labels.append(labels_dict['nothing'])

    

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

    model.add(Dense(29, activation = 'softmax'))

    

    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

    

    print("MODEL CREATED")

    model.summary()

    

    return model



def fit_model():

    model_hist = model.fit(X_train, Y_train, batch_size = 64, epochs = 6, validation_split = 0.1)

    return model_hist 
model = create_model()
curr_model_hist = fit_model()
def plot_accuracy(y):

    if(y == True):

        plt.plot(curr_model_hist.history['acc'])

        plt.plot(curr_model_hist.history['val_acc'])

        plt.legend(['train', 'test'], loc='lower right')

        plt.title('accuracy plot - train vs test')

        plt.xlabel('epoch')

        plt.ylabel('accuracy')

        plt.show()

    else:

        pass

    return



def plot_loss(y):

    if(y == True):

        plt.plot(curr_model_hist.history['loss'])

        plt.plot(curr_model_hist.history['val_loss'])

        plt.legend(['training loss', 'validation loss'], loc = 'upper right')

        plt.title('loss plot - training vs vaidation')

        plt.xlabel('epoch')

        plt.ylabel('loss')

        plt.show()

    else:

        pass

    return





plot_accuracy(True)

plot_loss(True)
evaluate_metrics = model.evaluate(X_test, Y_test)

print("\nEvaluation Accuracy =", evaluate_metrics[1],"\nEvaluation loss =",evaluate_metrics[0])
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
def give_predictions(test_data):

    predictions_classes = []

    for image in test_data:

        image = image.reshape(1,64,64,3)

        pred = model.predict_classes(image)

        predictions_classes.append(pred[0])

    return predictions_classes
predictions = give_predictions(test_images)
def get_labels_for_plot(predictions):

    predictions_labels = []

    for i in range(len(predictions)):

        for ins in labels_dict:

            if predictions[i] == labels_dict[ins]:

                predictions_labels.append(ins)

                break

    return predictions_labels
predictions_labels_plot = get_labels_for_plot(predictions)
predfigure = plt.figure(figsize = (14,14))

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
model.save('asl_model.h5')