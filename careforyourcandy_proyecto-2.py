# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os #File access

import time #measuring training time 

#Matplotlib and seaborn imports for graph plotting and image display 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sb

import itertools

#Sklearn imports

from sklearn.model_selection import train_test_split #Splitting dataset into train/dev sets

from sklearn.metrics import confusion_matrix #Confusion matrix 



#Keras imports

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential #To create sequential layers in a network

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D #Importing Dense (FC), Convolutional and Pooling Layers. 

from keras.layers.advanced_activations import LeakyReLU #Importing leaky relu

#Also imports Dropout regularizations and Flatten for FC Layers. 

from keras.optimizers import RMSprop, Adam #Optimizers

from keras.preprocessing.image import ImageDataGenerator #Data augmentation

#from keras.callbacks import ReduceLROnPlateau



#Checking directory structure

os.listdir('../input')

#Importing train and test datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#Getting the labels

Y_train = train["label"]

#Dropping the labels for X

X_train = train.drop("label", axis=1)



#Checking for NULL Values

print(train.isnull().any().describe())

print("===============================")

print(test.isnull().any().describe())
#Total examples

print(X_train.shape)

print(test.shape)

#Value counts

sb.countplot(Y_train)

Y_train.value_counts()


#Making a copy of X_train to compare normalized and non-normalized images

X_train_example = X_train

X_train_example = X_train_example.values.reshape(-1,28,28,1)



#Normalization 

X_train = X_train / 255.0

test = test / 255.0

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



#Comparison: We will compare how normalized images look compared to non-normalized. 

fig, comparison = plt.subplots(1, 2)

comparison[0].imshow(X_train[3][:,:,0]) 

comparison[1].imshow(X_train_example[3][:,:,0]) 
print("Previous shape of Y_train:")

print(Y_train.shape)

Y_train = to_categorical(Y_train, num_classes = 10)

print("New shape of Y_train:")

print(Y_train.shape)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size = 0.1)
datagen = ImageDataGenerator(

        rotation_range=10,  # rotates images by 10 deegres

        zoom_range = 0.10, # Randomly zoom image 

        width_shift_range=0.1,  # shifts images horizontally by a fraction of total width

        height_shift_range=0.1,  # same as above but vertically

        horizontal_flip=False,  # flip images

        vertical_flip=False)  #  flip images



#Note that we specifically use the False value for flips (Even though this is the default value provided by the function). This is to emphasize the fact that doing

#data augmentation by flipping numbers can bring problems, souch as mistaking a 6 for a 9, a 7 losing a shape that makes it a 7 for a human, etc. 



#Other thing to note is the usage of shifts. Even though filters in convolutional networks are very good at detecting shifts and data augmentation through shifts

#does not give the network that much "new learning material" we still use it to experiment. 

fig, comparison = plt.subplots(3, 2, sharex=True,sharey=True)



#EXAMPLE ONE: Slightly rotate images

comparison[0,0].imshow(X_train[0][:,:,0])

transform_dictionary = {"theta": 10}

generated_image = datagen.apply_transform(X_train[0], transform_dictionary)

comparison[0,1].imshow(generated_image[:,:,0], cmap='Greys')



#EXAMPLE TWO: Apply zoom in x 

transform_dictionary = {

    "zx": 0.8,

}

comparison[1,0].imshow(X_train[1][:,:,0])

generated_image = datagen.apply_transform(X_train[1], transform_dictionary)

comparison[1,1].imshow(generated_image[:,:,0], cmap='Greys')

#EXAMPLE THREE: Apply translation

transform_dictionary = {

    "tx": 4,

}

comparison[2,0].imshow(X_train[2][:,:,0])

generated_image = datagen.apply_transform(X_train[2], transform_dictionary)

comparison[2,1].imshow(generated_image[:,:,0], cmap='Greys')


model = Sequential() 

#MODEL ONE    

#First conv-conv-pool layer

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", input_shape=(28, 28, 1)))

model.add(LeakyReLU(0.1))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))

model.add(LeakyReLU(0.1))

model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

model.add(Dropout(0.25))

#Second conv-conv-pool layer

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same"))

model.add(LeakyReLU(0.1))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same"))

model.add(LeakyReLU(0.1))

model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

model.add(Dropout(0.25))

#Flatten

model.add(Flatten())

#First FC layer 

model.add(Dense(256))

model.add(LeakyReLU(0.1))

model.add(Dropout(0.5))

#Second FC layer 

model.add(Dense(10, activation="softmax"))





'''

model2 = Sequential() 

#MODEL TWO    

#First conv-conv-pool layer

model2.add(Conv2D(filters=16, kernel_size=(3,3), padding="same", input_shape=(28, 28, 1)))

model2.add(LeakyReLU(0.1))

model2.add(Conv2D(filters=32, kernel_size=(3,3), padding="same"))

model2.add(LeakyReLU(0.1))

model2.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

model2.add(Dropout(0.25))

#Second conv-conv-pool layer

model2.add(Conv2D(filters=48, kernel_size=(3,3), padding="same"))

model2.add(LeakyReLU(0.1))

model2.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))

model2.add(LeakyReLU(0.1))

model2.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

model2.add(Dropout(0.25))

#Flatten

model2.add(Flatten())

#First FC layer 

model2.add(Dense(256))

model2.add(LeakyReLU(0.1))

model2.add(Dropout(0.5))

#Second FC layer 

model2.add(Dense(10))

model2.add(LeakyReLU(0.1))

model.add(Dense(10, activation="softmax"))

'''

    
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

optimizer = Adam(lr=0.001)

model.compile(

    loss='categorical_crossentropy',  

    optimizer=optimizer,

    metrics=['accuracy']  

)

'''

model2.compile(

        loss='sparse_categorical_crossentropy',  

    optimizer=optimizer, 

    metrics=['accuracy']  

)

'''



model.summary()
#Set and labels: X_train, X_dev, Y_train, Y_dev

#Parameters

epochs = 30

batch_size = 64

#With data augmentation

start = time.time()

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                    validation_data = (X_dev, Y_dev),

                    epochs = epochs,

                    steps_per_epoch=X_train.shape[0] // batch_size

                   )

#We save the model on a history variable in order to evaluate the model later on. 

end = time.time()

print("Total time training was:")

print(end - start)

#Without data augmentation 

'''

model.fit(X_train, Y_train, 

          batch_size=batch_size,

          validation_data = (X_dev, Y_dev),

          epochs = epochs,

          #steps_per_epoch=X_train.shape[0] // batch_size

         )

'''
print(history.history.keys())
#Accuracy plot 

plt.plot(history.history['acc'], color="blue", label="Precisión de entrenamiento") #Paint blue line showing progressing train accuracy values

plt.plot(history.history['val_acc'], color="red", label="Precisión de dev") #Paint red line showing progressing val accuracy values

plt.legend(loc='best') #Place the legend where it doesn't overlap with the lines

plt.title('Accuracy vs epoch') #Set graph title

plt.ylabel('Precisión') #Set y label in graph

plt.xlabel('epoch') #Set x label in graph

plt.show()

#Loss plot

plt.plot(history.history['loss'], color='blue', label="Pérdida de entrenamiento")

plt.plot(history.history['val_loss'], color='red', label="Pérdida de dev")

plt.legend(loc='best')

plt.title('Pérdida vs epoch')

plt.ylabel('Pérdida')

plt.xlabel('epoch')

plt.show()
Y_dev_pred = model.predict(X_dev)

Y_dev_pred_value = np.argmax(Y_dev_pred, axis=1)

Y_dev_truth = np.argmax(Y_dev,axis = 1) 

confusion_mtx = confusion_matrix(Y_dev_truth, Y_dev_pred_value) 



def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
errors = (Y_dev_pred_value - Y_dev_truth != 0) #We compare true labels and predictions arrays. This will return true where the labels mismatch.

Y_pred_classes_errors = Y_dev_pred_value[errors] #Using python properties, we can pass a boolean array as an index for an array. This will return an array where all the values are true. 

Y_truth_errors = Y_dev_truth[errors]

print("Valores predecidos")

print(Y_pred_classes_errors)

print("Valores reales")

print(Y_truth_errors)

X_value_errors = X_dev[errors] #We extract the X values where there were errors. 

#We have a total of 25 errors. Let's display some of them. 



print(X_value_errors.shape)

i = 0



for number in range(4):

    rows = 2

    columns = 3 



    fig, ax = plt.subplots(rows, columns, sharex=True,sharey=True)

    plt.subplots_adjust(hspace = 0.4)

    for row in range(rows):

        for column in range(columns):

            ax[row, column].imshow(X_value_errors[i].reshape(28,28), cmap='Greys')

            ax[row,column].set_title("Predicted label :{}\nTrue label :{}".format(Y_pred_classes_errors[i],Y_truth_errors[i]))

            i += 1





#Y_pred_errors = Y_pred[errors]

#Y_true_errors = Y_true[errors]

#X_val_errors = X_val[errors]
predictions = model.predict(test)

results = np.argmax(predictions, axis = 1)

print(results)

results = pd.Series(results, name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("output.csv",index=False)