# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

import time



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from tensorflow import keras

from keras import losses

from keras import metrics

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



# Plot Confusion matrix function

def plot_confusion_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return None

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df            = pd.read_csv("../input/train.csv")

unlabeled_images_df = pd.read_csv("../input/test.csv")
#Para el conjunto de entrenamiento separa la imagen de su etiqueta de clase

train_images_df     = train_df.iloc[:,1:]

train_labels_df     = train_df.iloc[:,0:1]



#Convierte los datos de pandas df a numpy array

labeled_images      = train_images_df.values

labels              = train_labels_df.values

unlabeled_images    = unlabeled_images_df.values



#Convierte la etiqueta de clase en multicategorical

one_hot_labels      = keras.utils.to_categorical(labels, num_classes=10)



#Normaliza las imagenes de rango (0, 255) al rango (0,1) 

labeled_images      = labeled_images/255

unlabeled_images    = unlabeled_images/255



#Cambia la forma de las imagenes de vector size = 784 a matrix size = (28,28,1)

labeled_images      = np.reshape(labeled_images,(labeled_images.shape[0],28,28,1))

unlabeled_images    = np.reshape(unlabeled_images,(unlabeled_images.shape[0],28,28,1))
random_seed = 2



#Define el tamaño del conjunto con el que se va a entrenar la red

test_set_percentage       = 0.15

validation_set_percentage = 0.2



train_images,test_images,train_labels,test_labels                   = train_test_split(labeled_images, one_hot_labels, test_size = test_set_percentage, random_state=random_seed)

pretrain_images,validation_images,pretrain_labels,validation_labels = train_test_split(train_images, train_labels, test_size = validation_set_percentage, random_state=random_seed)
train_images_df.isnull().any().describe()
unlabeled_images_df.isnull().any().describe()
plt.title("Set sizes")

plot_image = [train_labels.shape[0],validation_labels.shape[0],test_labels.shape[0]]

plt.bar(["Train","Validation","Test"],height=plot_image)

plt.show()
size_of_img = (12,12)

fig         =plt.figure(figsize=(72,72))

ax          =fig.add_subplot(12,12,1)



plt.title("Train set")

plt.ylabel("Count")

plt.xlabel("Label")

plot_image = train_labels.sum(axis=0)

ax.bar([0,1,2,3,4,5,6,7,8,9],height=plot_image)

ax         =fig.add_subplot(12,12,2)



plt.title("Validation set")

plt.ylabel("Count")

plt.xlabel("Label")

plot_image = validation_labels.sum(axis=0)

ax.bar([0,1,2,3,4,5,6,7,8,9],height=plot_image)

ax         =fig.add_subplot(12,12,3)



plt.title("Test set")

plt.ylabel("Count")

plt.xlabel("Label")

plot_image = test_labels.sum(axis=0)

ax.bar([0,1,2,3,4,5,6,7,8,9],height=plot_image)

plt.show()
size_of_img = (int(np.sqrt(unlabeled_images.shape[1])),int(np.sqrt(unlabeled_images.shape[1])))

fig=plt.figure(figsize=(72,72))

for i in range(60):

    ax         =fig.add_subplot(12,12,i+1)

    plot_image = unlabeled_images[random.randint(0,1000),:,:,0]

    ax.imshow(plot_image)

plt.show()
data_gen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, 

                                                        samplewise_center=False, 

                                                        featurewise_std_normalization=False, 

                                                        samplewise_std_normalization=False, 

                                                        zca_whitening=False, 

                                                        zca_epsilon=1e-06, 

                                                        rotation_range=20, 

                                                        width_shift_range=0.05, 

                                                        height_shift_range=0.05, 

                                                        brightness_range=None, 

                                                        shear_range=0.05, 

                                                        zoom_range=0.05, 

                                                        channel_shift_range=0.0, 

                                                        fill_mode='nearest', 

                                                        cval=0.0, 

                                                        horizontal_flip=False, 

                                                        vertical_flip=False, 

                                                        rescale=0, 

                                                        preprocessing_function=None, 

                                                        data_format=None, 

                                                        validation_split=0.0, 

                                                        dtype=None)
data_gen.fit(pretrain_images)
adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=2, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)
model_1 = keras.Sequential()



model_1.add(keras.layers.Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same',activation='relu',input_shape=(28,28,1)))

model_1.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model_1.add(keras.layers.Dropout(0.25))



model_1.add(keras.layers.Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same',activation='relu'))

model_1.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model_1.add(keras.layers.Dropout(0.25))



model_1.add(keras.layers.Flatten())

model_1.add(keras.layers.Dense(128, activation='relu'))

model_1.add(keras.layers.Dense(10, activation='softmax'))
model_2 = keras.Sequential()



model_2.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model_2.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model_2.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model_2.add(keras.layers.Dropout(0.25))



model_2.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model_2.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model_2.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model_2.add(keras.layers.Dropout(0.25))



model_2.add(keras.layers.Flatten())

model_2.add(keras.layers.Dense(256, activation = "relu"))

model_2.add(keras.layers.Dropout(0.5))

model_2.add(keras.layers.Dense(10, activation = "softmax"))
model_3 = keras.Sequential()



model_3.add(keras.layers.Conv2D(filters = 32,kernel_size = (3,3),padding = 'Same',activation='relu',input_shape=(28,28,1)))

model_3.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model_3.add(keras.layers.Dropout(0.25))



model_3.add(keras.layers.Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same',activation='relu',input_shape=(28,28,1)))

model_3.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model_3.add(keras.layers.Dropout(0.25))



model_3.add(keras.layers.Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same',activation='relu'))

model_3.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model_3.add(keras.layers.Dropout(0.25))



model_3.add(keras.layers.Flatten())

model_3.add(keras.layers.Dense(128, activation='relu'))

model_3.add(keras.layers.Dense(10, activation='softmax'))
model_1.compile(optimizer = adam_optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])

model_2.compile(optimizer = adam_optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])

model_3.compile(optimizer = adam_optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
test_epochs  = 20
start          = time.time()

model1_history = model_1.fit_generator(data_gen.flow(pretrain_images, pretrain_labels, batch_size=32), epochs=test_epochs, validation_data=(validation_images,validation_labels), callbacks=[learning_rate_reduction])

end            = time.time()

model1_training_time = end - start
start          = time.time()

model2_history = model_2.fit_generator(data_gen.flow(pretrain_images, pretrain_labels, batch_size=32), epochs=test_epochs, validation_data=(validation_images,validation_labels), callbacks=[learning_rate_reduction])

end            = time.time()

model2_training_time = end - start
start          = time.time()

model3_history = model_3.fit_generator(data_gen.flow(pretrain_images, pretrain_labels, batch_size=32), epochs=test_epochs, validation_data=(validation_images,validation_labels), callbacks=[learning_rate_reduction])

end            = time.time()

model3_training_time = end - start
acc      = model1_history.history['acc']

val_acc  = model1_history.history['val_acc']

loss     = model1_history.history['loss']

val_loss = model1_history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
acc      = model2_history.history['acc']

val_acc  = model2_history.history['val_acc']

loss     = model2_history.history['loss']

val_loss = model2_history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
acc      = model3_history.history['acc']

val_acc  = model3_history.history['val_acc']

loss     = model3_history.history['loss']

val_loss = model3_history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
eval_model1 = model_1.evaluate(test_images, test_labels, batch_size=36)
eval_model2 = model_2.evaluate(test_images, test_labels, batch_size=36)
eval_model3 = model_3.evaluate(test_images, test_labels, batch_size=36)
models_accuracy = [eval_model1[1],eval_model2[1],eval_model3[1]]

training_times  = [model1_training_time,model2_training_time,model3_training_time]
# Predict the values from the validation dataset

model1_pred = model_1.predict(test_images)

model2_pred = model_2.predict(test_images)

model3_pred = model_3.predict(test_images)

# Convert predictions classes to one hot vectors 

model1_pred_classes = np.argmax(model1_pred,axis = 1) 

model2_pred_classes = np.argmax(model2_pred,axis = 1) 

model3_pred_classes = np.argmax(model3_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(test_labels, axis = 1) 
# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, model1_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, model2_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, model3_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
size_of_img = (12,12)

fig         =plt.figure(figsize=(72,72))



ax          =fig.add_subplot(12,12,1)

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Model")

plot_image = models_accuracy

ax.bar([1,2,3],height=plot_image)



ax         =fig.add_subplot(12,12,2)

plt.title("Training time")

plt.ylabel("Time")

plt.xlabel("Model")

plot_image = training_times

ax.bar([1,2,3],height=plot_image)



plt.show()