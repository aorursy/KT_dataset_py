# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#reading train dataset

train = pd.read_csv("../input/fashion-mnist_train.csv")

print(train.shape)
# let's look at first five train samples

train.head()
train.iloc[0].value_counts()
# reading test dataset

test = pd.read_csv("../input/fashion-mnist_test.csv")

print(test.shape)
# let's look at first five test samples

test.head()
# put labels into Y_train variable

Y_train = train["label"].values

# Drop 'label' column

X_train = train.drop(labels = ["label"], axis = 1)

X_train.head()
# put labels into Y_test variable

Y_test = test["label"].values

# Drop 'label' column

X_test = test.drop(labels = ["label"], axis = 1)

X_test.head()
plt.figure(figsize=(14,8))

sns.countplot(Y_train, palette="icefire")

plt.title("Number of classes")

plt.show()
plt.figure(figsize=(14,8))

sns.countplot(Y_test, palette="icefire")

plt.title("Number of classes")

plt.show()
# plot some samples

plt.figure(figsize=(4,4))

plt.title(Y_train[0])

plt.imshow(X_train.values.reshape(-1,28,28)[0],cmap="gray")
# plot some samples

plt.figure(figsize=(4,4))

plt.title(Y_train[17])

plt.imshow(X_train.values.reshape(-1,28,28)[17],cmap="gray")
plt.figure(figsize = (14,8))



for i in range(10):

    plt.subplot(2, 5, i+1)

    img = train[train.label == i].iloc[0, 1:].values

    img = img.reshape((28,28))

    plt.imshow(img, cmap='gray')

    plt.title("Class: " + str(i))

    plt.axis('off')

    

plt.show()
# Normalize the data

X_train = X_train / 255.0

X_test = X_test / 255.0

print("X_train shape: ",X_train.shape)

print("X_test shape: ",X_test.shape)
X_train.head()
X_test.head()
# Reshaping

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

print("X_train shape: ",X_train.shape)

print("X_test shape: ",X_test.shape)
# Label Encoding (be careful! run just once!)

from keras.utils.np_utils import to_categorical 



# convert to one-hot-encoding(one hot vectors)

Y_train = to_categorical(Y_train, num_classes = 10)

# convert to one-hot-encoding(one hot vectors)

Y_test = to_categorical(Y_test, num_classes = 10)



print(Y_train.shape)

print(Y_test.shape)
# Split the train and the validation set for the fitting

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)

print("x_train shape: ",x_train.shape)

print("x_val shape: ",x_val.shape)

print("y_train shape: ",y_train.shape)

print("y_val shape :",y_val.shape)
# Some examples

plt.imshow(x_train[4].reshape(28,28),cmap="gray")

plt.axis("off")

plt.show()
import warnings

warnings.filterwarnings('ignore')



from sklearn.metrics import confusion_matrix



from keras.models import Sequential, model_from_json

from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint



model = Sequential()



#1. LAYER

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Activation("relu"))



#2. LAYER

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(Activation("relu"))



model.add(MaxPool2D(pool_size=(2, 2)))



#3. LAYER

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(Activation("relu"))



#4. LAYER

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(Activation("relu"))



model.add(MaxPool2D(pool_size=(2, 2)))



#FULLY CONNECTED LAYER

model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.25))



#OUTPUT LAYER

model.add(Dense(10, activation='softmax'))
model.summary()
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# # Define the optimizer

# optimizer = RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 50 # for better result increase the epochs

batch_size = 100
# Data Augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.1,  # randomly rotate images in the range

        zoom_range = 0.1, # Randomly zoom image

        width_shift_range=0.1,  # randomly shift images horizontally

        height_shift_range=0.1,  # randomly shift images vertically

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(x_train)
# # save the best weights

# checkpointer = ModelCheckpoint(filepath="../yourPath/fashion_mnist_model.h5", verbose=1, save_best_only=True)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                              shuffle=True, #veriler random gelip eğitilir

                              epochs=epochs, validation_data = (x_val, y_val),

                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size)

#                               callbacks=[checkpointer]) #we save the best weights with checkpointer
# # save model to json

# model_json = model.to_json() #fashion_mnist_model.h5 - I saved the file in JSON format.

# with open("../yourPath/fashion_mnist_model.json", "w") as json_file:

#     json_file.write(model_json)
# # load the best weights which we saved

# model_best = load_model("../yourPath/fashion_mnist_model.h5")
plt.figure(figsize=(14,5))

plt.subplot(1, 2, 1)

plt.suptitle('Train Results', fontsize=10)

plt.xlabel("Number of Epochs")

plt.ylabel('Loss', fontsize=16)

plt.plot(history.history['loss'], color='b', label='Training Loss')

plt.plot(history.history['val_loss'], color='r', label='Validation Loss')

plt.legend(loc='upper right')



plt.subplot(1, 2, 2)

plt.ylabel('Accuracy', fontsize=16)

plt.plot(history.history['acc'], color='green', label='Training Accuracy')

plt.plot(history.history['val_acc'], color='orange', label='Validation Accuracy')

plt.legend(loc='lower right')

plt.show()
print('Train accuracy of the model: ',history.history['acc'][-1])
print('Train loss of the model: ',history.history['loss'][-1])
print('Validation accuracy of the model: ',history.history['val_acc'][-1])
print('Validation loss of the model: ',history.history['val_loss'][-1])
score = model.evaluate(X_test,Y_test,verbose=0)

print("Test Loss:",score[0])

print("Test Accuracy:",score[1])
print(X_test.shape)

plt.imshow(X_test[100].reshape(28,28),cmap="gray")

plt.axis("off")

plt.show()
trueY = Y_test[100]

img = X_test[100]

test_img = img.reshape(1,28,28,1)



preds = model.predict_classes(test_img)

prob = model.predict_proba(test_img)



print("trueY: ",np.argmax(trueY))#içlerinden en yüksek olan değeri seçer

print("Preds: ",preds)

print("Prob: ",prob)
# image_path = "../yourPath/test_image.jpg"



# test_image_orjinal = image.load_img(image_path) # orjinal renkli görüntü



# test_image = image.load_img(image_path, target_size=(48,48), grayscale=True)

# test_data = image.img_to_array(test_image)

# test_img = test_data.reshape(1,48,48,1)



# preds = model_best.predict_classes(test_img)

# prob = model_best.predict_proba(test_img)



# print("Preds: ",preds)

# print("Prob: ",prob)
Y_pred = model.predict(X_test)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

Y_true = np.argmax(Y_test, axis = 1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.1, cmap = "gist_yarg_r", linecolor="black", fmt='.0f', ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
for i in range(len(confusion_mtx)):

    print("Class:",str(i))

    print("Number of Wrong Prediction:", str(sum(confusion_mtx[i])-confusion_mtx[i][i]), "out of 1000")

    print("Percentage of True Prediction: {:.2f}%".format(confusion_mtx[i][i] / 10))

    print("***********************************************************")