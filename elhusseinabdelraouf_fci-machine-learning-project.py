import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# Settings:

img_size = 64

num_class = 10

validation_size = 0.2

test_size = 0.2



# Import Needed Packages for data preperation

from os import listdir

from imageio import imread

from PIL import Image

from keras.utils import to_categorical



# Helper Function To Read The Dataset

def get_img(data_path, grayscale_images):

    # Getting image array from path:

    img = imread(data_path, as_gray=grayscale_images)

    img = np.array(Image.fromarray(img).resize((img_size, img_size)))

    return img



def get_dataset(dataset_path='Dataset', grayscale_images=False):

    labels = listdir(dataset_path) # Geting labels

    labels.sort()

    X = []

    Y = []

    for i, label in enumerate(labels):

        datas_path = dataset_path+'/'+label

        for data in listdir(datas_path):

            img = get_img(datas_path+'/'+data, grayscale_images)

            X.append(img)

            Y.append(i)

    # Create dateset:

    X = np.array(X).astype('float32')

    

    # Normalization Step     

    if grayscale_images:

        X = X/255.

    else:        

        imagesAvg = np.average(X)

        X = (X - imagesAvg)/255.



    Y = np.array(Y).astype('float32')

    Y = to_categorical(Y, num_class)

    return X, Y,
XGray, YGray = get_dataset(dataset_path='../input/sign-language-datasen/Dataset', grayscale_images=True)
plt.subplot(1, 2, 1)

plt.imshow(XGray[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(XGray[900].reshape(img_size, img_size))

plt.axis('off')
XRGB, YRGB = get_dataset(dataset_path='../input/sign-language-datasen/Dataset', grayscale_images=False)
plt.subplot(1, 2, 1)

plt.imshow(XRGB[260].reshape(img_size, img_size, 3))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(XRGB[900].reshape(img_size, img_size, 3))

plt.axis('off')
# import needed pacakges for data split

from sklearn.model_selection import train_test_split
# Split Gray Dataset

XGray, xGray_test, YGray, yGray_test = train_test_split(XGray, YGray, test_size = 0.20, random_state = 42)

xGray_train, xGray_validation, yGray_train, yGray_validation = train_test_split(XGray, YGray, test_size = 0.25, random_state = 42)



xGray_train = xGray_train.reshape(-1,64,64,1)

xGray_test = xGray_test.reshape(-1,64,64,1)

xGray_validation = xGray_validation.reshape(-1,64,64,1)



print(xGray_train.shape)

print(yGray_train.shape)



print(xGray_test.shape)

print(yGray_test.shape)



print(xGray_validation.shape)

print(yGray_validation.shape)
# Split RGB Dataset

XRGB, xRGB_test, YRGB, yRGB_test = train_test_split(XRGB, YRGB, test_size = 0.20, random_state = 42)

xRGB_train, xRGB_validation, yRGB_train, yRGB_validation = train_test_split(XRGB, YRGB, test_size = 0.25, random_state = 42)



xRGB_train = xRGB_train.reshape(-1,64,64,3)

xRGB_test = xRGB_test.reshape(-1,64,64,3)

xRGB_validation = xRGB_validation.reshape(-1,64,64,3)



print(xRGB_train.shape)

print(yRGB_train.shape)



print(xRGB_test.shape)

print(yRGB_test.shape)



print(xRGB_validation.shape)

print(yRGB_validation.shape)
# Import Needed Libraries to create CNN



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import KFold
# CNN Settings

epochs = 30

batch_size = 20



annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

datagen = ImageDataGenerator()



# helper evaluation functions

from keras import backend as K

def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def fscore(y_true, y_pred):

    precision_value = precision(y_true, y_pred)

    recall_value = recall(y_true, y_pred)

    return 2*((precision_value*recall_value)/(precision_value+recall_value+K.epsilon()))





def plot_history(history):

    # Plot the loss and accuracy curves for training and validation 

    plt.plot(history.history['val_loss'], color = 'b', label = "validation loss")

    plt.title("Test Loss")

    plt.xlabel("Number of Epochs")

    plt.ylabel("Loss")

    plt.legend()

    plt.show()



    accuracy = history.history['acc']

    val_accuracy = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochsRange = range(len(accuracy))





    plt.plot(epochsRange, accuracy, 'bo', label = 'Training accuracy')

    plt.plot(epochsRange, val_accuracy, 'b', label = 'Validation accuracy')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.show()





    plt.plot(epochsRange, loss, 'bo', label = 'Training loss')

    plt.plot(epochsRange, val_loss, 'b', label = 'Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.show()

    

    

def train_and_report(modelBuilder, XTrain, YTrain, XValidation, YValidation, XTest, YTest):    

    # K Fold Cross Validation

    kf = KFold(n_splits=2)

    i = 1

    for train, test in kf.split(XTrain, YTrain):

        print("Fold " + str(i))

        model = modelBuilder()

        model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics = ["acc", fscore, precision, recall])

        # fit the model

        history = model.fit_generator(datagen.flow(XTrain[train], YTrain[train], batch_size = batch_size), 

                                  epochs = epochs, 

                                  validation_data = (XTrain[test], YTrain[test]), 

                                  steps_per_epoch = XTrain[train].shape[0] // batch_size,

                                  verbose=0

                                 )

        

        plot_history(history)

        i += 1

        



    print("Full Data")

    model = modelBuilder()

    model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics = ["acc", fscore, precision, recall])

    history = model.fit_generator(datagen.flow(XTrain, YTrain, batch_size = batch_size), 

                                  epochs = epochs, 

                                  validation_data = (XValidation, YValidation), 

                                  steps_per_epoch = XTrain.shape[0] // batch_size,

                                  verbose=0

                                 )

    plot_history(history)



    final_loss, final_acc, final_fscore, final_precision, final_recall, *t = model.evaluate(XTest, YTest, verbose = 0)

    print("Final loss: {0:.4f}, Final accuracy: {1:.4f}, Final fscore: {2:.4f}, Final precision: {3:.4f}, Final recall: {4:.4f}".format(final_loss, final_acc, final_fscore, final_precision, final_recall))
def GrayBuilder_1():

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))





    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))



    return model



train_and_report(GrayBuilder_1, xGray_train, yGray_train, xGray_validation, yGray_validation, xGray_test, yGray_test)
def GrayBuilder_2():

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))



    return model



train_and_report(GrayBuilder_2, xGray_train, yGray_train, xGray_validation, yGray_validation, xGray_test, yGray_test)
def GrayBuilder_3():

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))



    return model



train_and_report(GrayBuilder_3, xGray_train, yGray_train, xGray_validation, yGray_validation, xGray_test, yGray_test)
def GrayBuilder_4():

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))



    return model



train_and_report(GrayBuilder_4, xGray_train, yGray_train, xGray_validation, yGray_validation, xGray_test, yGray_test)
def RGBBuilder_1():

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,3)))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))



    return model



train_and_report(RGBBuilder_1, xRGB_train, yRGB_train, xRGB_validation, yRGB_validation, xRGB_test, yRGB_test)
def RGBBuilder_2():

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,3)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))





    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))



    return model



train_and_report(RGBBuilder_2, xRGB_train, yRGB_train, xRGB_validation, yRGB_validation, xRGB_test, yRGB_test)
def RGBBuilder_3():

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,3)))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 32, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 32, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))

    model.add(Dropout(0.25))



    

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    

    return model



train_and_report(RGBBuilder_3, xRGB_train, yRGB_train, xRGB_validation, yRGB_validation, xRGB_test, yRGB_test)
def RGBBuilder_4():



    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu', input_shape = (64,64,3)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 64, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 32, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 32, kernel_size = (4,4), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))



    return model



train_and_report(RGBBuilder_4, xRGB_train, yRGB_train, xRGB_validation, yRGB_validation, xRGB_test, yRGB_test)