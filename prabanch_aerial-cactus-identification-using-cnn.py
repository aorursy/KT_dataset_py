import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from keras.optimizers import SGD

import cv2

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation



# from resnets_utils import *



from keras.models import load_model



from keras.utils import np_utils



from keras import applications



from keras import optimizers

from keras.models import Sequential,Model,load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
import os

# print(os.listdir("../input/test/test/"))
path = "../input/"

train=pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'sample_submission.csv')
test.head()
# As it is a multi-class classification problem (3 classes), we will one-hot encode the target variable.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

train['has_cactus_en'] = labelencoder.fit_transform(train['has_cactus'])







print(train['has_cactus'].value_counts())

print(train['has_cactus'].value_counts())





y=train['has_cactus']
train.has_cactus.value_counts()
def load_images(path, df):

    train_image = []

    for i in tqdm(range(df.shape[0])):

        try:

#             print(df_image_path+df['id'][i])

            img = cv2.imread((path+df['id'][i]))

            img = cv2.resize(img, (64,64))



#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



            img = image.img_to_array(img)

            img = img/255

            train_image.append(img)

        except OSError:

            print(df['id'][i])

    image_array = np.array(train_image)   

    return image_array

        

            

train_image_path = '../input/train/train/'

test_image_path  = '../input/test/test/'



X = load_images(train_image_path,train)

test_images = load_images(test_image_path, test)

# test_image = []

# for i in tqdm(range(test.shape[0])):

# # for i in range(0,1):

# #     try:

#     img = cv2.imread((image_path+test['image'][i]))

#     img = cv2.resize(img, (200,120))

# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     img = image.img_to_array(img)

#     img = img/255

    

#     test_image.append(img)

# #     except OSError:

# #         print(train['file_path'][i])

# test_images = np.array(test_image)
import seaborn as sns

sns.countplot(train['has_cactus'])
import numpy as np

from sklearn.utils.class_weight import compute_class_weight



class_weights = compute_class_weight('balanced', np.unique(y), y)

class_weights
# Normalize the data

# X = X / 255.0

# test = test / 255.0



# Step 4: Creating a validation set from the training data.



X_train, X_test, y_train, y_test_class = train_test_split(X, y, random_state=42, test_size=0.2)



y_train = pd.get_dummies(y_train)

y_test = pd.get_dummies(y_test_class)

# Step 5: Define the model structure.



# We will create a simple architecture with 2 convolutional layers, one dense hidden layer and an output layer.



model = Sequential()

# model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu',input_shape=(200,200,3)))

# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



model.add(Conv2D(32, (3, 3), padding="same",  activation='relu',input_shape=(64,64,3)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))



# (CONV => RELU) * 2 => POOL

model.add(Conv2D(64, (3, 3), padding="same", activation='relu' ))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

          

          

# (CONV => RELU) * 2 => POOL

model.add(Conv2D(128, (3, 3), padding="same", activation='relu' ))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding="same", activation='relu' ))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

          

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.50))



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(2, activation='softmax'))
model.summary()
# Next, we will compile the model we’ve created.

opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

# Data Augmentation

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



datagen = ImageDataGenerator(

        rotation_range=10,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



datagen.fit(X_train)

print('augmentation')
# Step 6: Training the model.



# In this step, we will train the model on the training set images and validate it using, you guessed it, the validation set.

batch_size = 200

# results  = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))



results  = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), epochs=1, 

                              steps_per_epoch=X_train.shape[0] // batch_size, 

                              validation_data=(X_test, y_test),

                             class_weight = class_weights)

print(' class weight')
type(model.history)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(results.history['loss'], color='b', label="Training loss")

ax[0].plot(results.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(results.history['acc'], color='b', label="Training accuracy")

ax[1].plot(results.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Step 7: Find Accuracy

final_loss, final_acc = model.evaluate(X_test, y_test, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))


# Look at confusion matrix 

import itertools

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

    plt.xticks(tick_marks, classes , rotation=45)

    plt.yticks(tick_marks, classes)





    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict_classes(X_test) 



# compute the confusion matrix

confusion_mtx = confusion_matrix(y_test_class, Y_pred) 

confusion_mtx

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(0,2)) 
base_cnn_pred  = model.predict_classes(test_images)
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (64,64,3))



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.2)(x)

predictions = Dense(2, activation= 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

from keras.optimizers import SGD, Adam

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

adam = Adam(lr=0.0001)

model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])



# model.fit(X_train, Y_train, epochs = 100, batch_size = 64)



model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), epochs=100, 

                              steps_per_epoch=X_train.shape[0] // batch_size, 

                              validation_data=(X_test, y_test),

                             class_weight = class_weights)
# Predict the values from the validation dataset

Y_pred = model.predict(X_test) 

type(Y_pred)

Y_pred = Y_pred.argmax(axis=-1)





# # compute the confusion matrix

confusion_mtx = confusion_matrix(y_test_class, Y_pred) 

confusion_mtx

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(0,2)) 
resnet_pred =  model.predict(test_images)

# resnet_pred = resnet_pred.argmax(axis=-1)





# final_predictions =  (base_cnn_pred + resnet_pred) / 2

# final_predictions = np.where(final_predictions >= 0.5, 1, 0)

test['has_cactus'] = resnet_pred



test[['id','has_cactus']]

test.to_csv('output.csv',  index=False)