#Easy data manipulation

import pandas as pd

import numpy as np



#Plotting

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

import matplotlib.pyplot as plt



#Who likes warnings anyway?

import warnings

warnings.filterwarnings('ignore')



#Sklearn stuffs

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold



#Keras stuffs

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D, AveragePooling2D

from keras.optimizers import Adadelta



#Math

from math import ceil



#Time to measure runtime

from time import time
#Tells us which class is associated with which modern hiragana

classmap = pd.read_csv('../input/kmnist_classmap.csv')



#The set is already split in a training and a testing set, and has separated labels

train_images = np.load('../input/kmnist-train-imgs.npz')['arr_0']

test_images = np.load('../input/kmnist-test-imgs.npz')['arr_0']

train_labels = np.load('../input/kmnist-train-labels.npz')['arr_0']

test_labels = np.load('../input/kmnist-test-labels.npz')['arr_0']
print("KMNIST train shape:", train_images.shape)

print("KMNIST test shape:", test_images.shape)

print("KMNIST train shape:", train_labels.shape)

print("KMNIST test shape:", test_labels.shape)



#Adding hiragana's romaji to the class map for convenient display

romaji = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

classmap['romaji'] = romaji

print("\nKMNIST class map shape:", classmap.shape)

print('\nClass map:\n', classmap)
Classes_proportion_train = [len(train_images[np.where(train_labels == i)])/len(train_images)*100 for i in range(len(classmap))]

Classes_proportion_test = [len(test_images[np.where(test_labels == i)])/len(test_images)*100 for i in range(len(classmap))]



print("----- PROPORTION OF CLASSES IN TRAINING SET -----\n")

for i in range(len(classmap)):

    print("Proportion of class {0}: {1}%". format(i, Classes_proportion_train[i]))



print("\n----- PROPORTION OF CLASSES IN TEST SET -----\n")

for i in range(len(classmap)):

    print("Proportion of class {0}: {1}%". format(i, Classes_proportion_test[i]))

figure = plt.figure(figsize=(15,5))

figure.suptitle('Labeled hiragana examples from the data set', fontsize=16)

for lab in range(len(classmap)):

    images = train_images[np.where(train_labels == lab)]

    labels = train_labels[np.where(train_labels == lab)]

    for inst in range(3):

        

        #Make a grid of 10x3. Each line will receive the 3 first example of one of the 10 classes

        plt.subplot(3,10,1 + lab + (inst * 10)) #Be careful with the subplot index, it begins at 1, not 0

        

        #Plot image with label as title

        plt.imshow(images[inst], cmap=plt.cm.Greys) #We use grayscale for readability

        plt.title(labels[inst]) 

        #We can't display the computer-version if the modern hiragana as the plots' title

        #matplotlib doesn't seem to support these characters

        

        #Formatting: no grid, no ticks

        plt.grid(False)

        plt.xticks(ticks=[])

        plt.yticks(ticks=[])

            

plt.show()
flat_image_train = np.reshape(train_images, (60000, -1))

flat_image_test = np.reshape(test_images, (10000, -1))
ss = StandardScaler()

flat_image_train = ss.fit_transform(flat_image_train)

flat_image_test = ss.transform(flat_image_test)
start = time()

knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)

knn.fit(flat_image_train, train_labels)

y_predicted = knn.predict(flat_image_test)

end = time()



print(classification_report(test_labels, y_predicted))

print("kNN accuracy: {0:.2f}%".format(knn.score(flat_image_test, test_labels)))

print("kNN took {0:.2f} seconds to perform fit and predict.".format(end-start))
x_train = np.reshape(train_images, (60000, 28, 28,1))  

y_train = keras.utils.to_categorical(train_labels, num_classes=len(classmap))

x_test = np.reshape(test_images, (10000, 28, 28,1))

y_test = keras.utils.to_categorical(test_labels, num_classes=len(classmap))
def GetMyConvNet():

    model = Sequential()



    model.add(Conv2D(32, (3, 3), strides=(1,1), padding="same", input_shape=(28, 28,1)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(Conv2D(32, (3, 3), strides=(1,1), padding="same"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling2D(pool_size=(2, 2))) 

    #model.add(Dropout(0.25))



    model.add(Conv2D(62, (3, 3), strides=(1,1), padding="same"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(62, (3, 3), strides=(1,1), padding="same"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))



    model.add(Conv2D(96, (3, 3), strides=(1,1), padding="same"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(Conv2D(96, (3, 3), strides=(1,1), padding="same"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(AveragePooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(384, activation='relu'))

    model.add(Dense(192, activation='relu'))

    #model.add(Dropout(0.5))

    model.add(Dense(10))

    model.add(BatchNormalization())

    model.add(Activation('softmax'))



    adadelta = Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])



    return model
GetMyConvNet().summary()
model = GetMyConvNet()



start=time()



history = model.fit(x_train, y_train, batch_size=256, epochs=100)

score = model.evaluate(x_test, y_test)



elapsed_time = time() - start
print("Accuracy on test set: {0:.2f}%\nLoss: {1:.2f}\nTime elapsed: {2:.2f} seconds".format(score[1]*100, score[0], elapsed_time))
figure = plt.figure(figsize=(12,3))

plt.subplot(1,2,1)



#Accuracy

plt.plot(history.history['acc'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Accuracy over time\n')

plt.legend(['Train Accuracy','Test Accuracy'])

#plt.show()



plt.subplot(1,2,2)



#Loss

plt.plot(history.history['loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Loss over time\n')

plt.legend(['Train Loss','Test Loss'])



plt.show()
#Find missclassified from the test set and put their index in a list

predicted = model.predict(x_test)

preds = [np.argmax(predicted[x]) for x in range(len(predicted))]

missclassified= [i for i in range(len(predicted)) if preds[i]!=test_labels[i]]



#Images, Labels and Predictions for missclassified images

images = test_images[[i for i in missclassified]]

labels = test_labels[[i for i in missclassified]]

missed_pred = [preds[i] for i in missclassified]



figure = plt.figure(figsize=(20, 6*ceil(len(images)/10)))



for inst in range(len(images)):

       

    #Make a grid 

    plt.subplot(ceil(len(images)/5), 10, 2*inst+1)

       

    #Plot image with predicted and actual labels as title

    plt.imshow(images[inst], cmap=plt.cm.Greys)

    plt.title("Predicted: {0} ({1})\nActual: {2} ({3})".format(missed_pred[inst],\

                                                               classmap[classmap['index']==missed_pred[inst]]['romaji'].values[0],\

                                                               labels[inst],\

                                                               classmap[classmap['index']==labels[inst]]['romaji'].values[0])) 

    

    #Formatting: no grid, no tick

    plt.grid(False)

    plt.xticks(ticks=[])

    plt.yticks(ticks=[])

            



    plt.subplot(ceil(len(images)/5), 10, 2*inst+2)

    plt.bar(range(10), predicted[[i for i in missclassified]][inst])

    #Formatting: no grid, no tick

    plt.grid(False)

    plt.xticks(range(10), range(10))

    plt.yticks(ticks=[])



print("----- Mislabeled hiraganas from the test set -----")

plt.show()

#Concatenate training and test set and labels.

x_full = np.reshape(np.concatenate((x_train, x_test), axis=0), (70000, -1)) #Flatten images to allow direct use of StratifiedKFold

y_full = np.concatenate((train_labels, test_labels), axis=0)



kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=12)

cvscores = []

cvtimes = []

fold=0



for train, test in kfold.split(x_full, y_full):

    

    #Training and Test sets and labels for the fold

    x_full_train = np.reshape(x_full[train], (len(x_full[train]), 28, 28, 1)) #Restore the images as 28x28x1

    y_full_train = keras.utils.to_categorical(y_full[train], num_classes=len(classmap)) #One-hot encoding of labels

    x_full_test = np.reshape(x_full[test], (len(x_full[test]), 28, 28, 1)) #Restore the images as 28x28x1

    y_full_test = keras.utils.to_categorical(y_full[test], num_classes=len(classmap))#One-hot encoding of labels

    

    start=time()

    fold+=1

    

    model = GetMyConvNet()



    model.fit(x_full_train, y_full_train, batch_size=256, epochs=100, verbose=0)

    scores = model.evaluate(x_full_test, y_full_test, verbose=0)

    

    elapsed_time = time() - start

    print("Accuracy for fold n°{0}: {1:.2f}% ({2:.2f} seconds)".format(fold,scores[1]*100, elapsed_time))

    

    cvscores.append(scores[1] * 100)

    cvtimes.append(elapsed_time)

    

print("\n\nMean accuracy on {0} folds: {1:.2f}% (+/- {2:.2f})\nTotal elapsed time for {0}-fold validation: {3:.2f} seconds".format(fold, np.mean(cvscores), np.std(cvscores), np.sum(cvtimes)))