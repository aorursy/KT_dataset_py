import numpy as np

import pandas as pd

import os

import keras

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras import regularizers

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input/bisindotest/bisindo_test"))
train_dir = "../input/bisindo2/Bisindo"
def load_unique():

    size_img = 64,64 

    images_for_plot = []

    labels_for_plot = []

    for folder in os.listdir(train_dir):

        for file in os.listdir(train_dir + '/' + folder):

            filepath = train_dir + '/' + folder + '/' + file

            image = cv2.imread(filepath)

            final_img = cv2.resize(image, size_img)

            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)

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

    plt.imshow(image,cmap='Greys')

    plt.title(label)

    return



image_index = 0

row = 5

col = 5

for i in range(1,(row*col+1)):

    plot_images(fig, images_for_plot[image_index], labels_for_plot[image_index], row, col, i)

    image_index = image_index + 1

plt.show()
labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,

                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,

                   'Z':25}



def load_data():

    """

    Loads data and preprocess. Returns train and test data along with labels.

    """

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

    

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.12)

    

    print()

    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)

    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)

    

    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = load_data()
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

model.add(Dense(26, activation = 'softmax'))

    

model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

    

print("MODEL CREATED")

model.summary()

    

es = EarlyStopping(monitor='val_loss', verbose=1, patience=2)

    
from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

#kf= StratifiedKFold(n_splits=3)

kf = KFold(n_splits=3)
def get_score (model, x_train, x_test, y_train, y_test):

    curr_model_hist= model.fit(x_train,y_train, batch_size = 64, callbacks=[es],validation_data=(x_test,y_test),epochs=5) 

    test_score = model.evaluate(x_test,y_test)

    train_score = model.evaluate(x_train,y_train)

    plotHistory(curr_model_hist)

    return train_score,test_score
def plotHistory(curr_model_hist):

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
scores = []



for train_index, test_index in kf.split(X_train):

    x_train, x_test, y_train, y_test = X_train[train_index], X_train[test_index], Y_train[train_index], Y_train[test_index]

    scores.append(get_score(model,x_train,x_test,y_train,y_test))
scores
total_train_loss = 0

total_train_acc = 0

total_test_loss = 0

total_test_acc = 0

for i in range (3):

    total_train_loss = total_train_loss+scores[i][0][0]

    total_train_acc =total_train_acc+scores[i][0][1]

    total_test_loss =total_test_loss+scores[i][1][0]

    total_test_acc =total_test_acc+scores[i][1][1]

print(total_train_acc/3)

print(total_train_loss/3)

print(total_test_acc/3)

print(total_test_loss/3)
evaluate_metrics = model.evaluate(X_test, Y_test)

print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
def evaluate_F1(model,Y_test,X_test):

    from sklearn.metrics import f1_score

    y_true = [np.where(r==1)[0][0] for r in Y_test]

    classes = model.predict_classes(X_test)

    return f1_score(y_true,classes,average=None)
evaluate_F1(model,Y_test,X_test)
model.save('MODEL A.h5')
from keras.models import load_model
source_model = load_model("../input/asl-classifier-using-keras/ASL3")

source_model.summary()
new_model = Sequential()

new_model = source_model

new_model.pop()

new_model.add(Dense(26))
for i in range (8):

    new_model.layers[i+1].trainable = False

new_model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

new_model.summary()
scores = []



for train_index, test_index in kf.split(X_train):

    x_train, x_test, y_train, y_test = X_train[train_index], X_train[test_index], Y_train[train_index], Y_train[test_index]

    scores.append(get_score(new_model,x_train,x_test,y_train,y_test))
total_train_loss = 0

total_train_acc = 0

total_test_loss = 0

total_test_acc = 0

for i in range (3):

    total_train_loss = total_train_loss+scores[i][0][0]

    total_train_acc =total_train_acc+scores[i][0][1]

    total_test_loss =total_test_loss+scores[i][1][0]

    total_test_acc =total_test_acc+scores[i][1][1]

print(total_train_acc/3)

print(total_train_loss/3)

print(total_test_acc/3)

print(total_test_loss/3)
evaluate_metrics = new_model.evaluate(X_test, Y_test)

print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
evaluate_F1(new_model,Y_test,X_test)
new_model.save( 'MODEL B.h5')