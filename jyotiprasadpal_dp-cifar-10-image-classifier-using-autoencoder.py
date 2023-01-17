import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, MaxPool2D, Flatten, BatchNormalization

from keras import regularizers

from keras.models import Model, Sequential

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.optimizers import SGD, Adam, RMSprop, Adadelta

import keras.backend as K

from keras.objectives import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils, to_categorical

from keras.datasets import cifar10



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
(X_train_images, y_train_labels), (X_test_images, y_test_labels) = cifar10.load_data()



print('X_train_images.shape: ', X_train_images.shape)

print('X_test_images.shape: ', X_test_images.shape)

print('y_train_labels.shape: ', y_train_labels.shape)

print('y_test_labels.shape: ', y_test_labels.shape)
## function used for visualizing the predicted and true labels of test data

def show_test(m, X_test, y_test, d):

    plt.figure(figsize =(40,8))

    for i in range(5):

        ax = plt.subplot(1, 5, i+1)

        test_image = np.expand_dims(d[1810*i+5], axis=0)

        test_result = m.predict(test_image)

        plt.imshow(X_test[1810*i+5])

        index = np.argsort(test_result[0,:])

        plt.title("Pred:{}, True:{}".format(dict[index[9]], dict[y_test[1810*i+5][0]]))

    plt.show()



## function used for creating a classification report and confusion matrix

def report(predictions, y_test):

    cm=confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

    

    print("Classification Report:\n")

    cr=classification_report(y_test.argmax(axis=1),

                                predictions.argmax(axis=1), 

                                target_names=list(dict.values()))

    print(cr)

    

    #plot confusion_matrix

    plt.figure(figsize=(8,6))

    sns.heatmap(cm, annot=True, xticklabels = list(dict.values()), yticklabels = list(dict.values()), fmt="d")

    

## function used for visualizing original and reconstructed images of the autoencoder model

def showOrigAndReconstructedImages(orig_imgs, decoded_imgs, img_width, img_height, channels, num=10):

    n = num

    plt.figure(figsize=(20, 4))



    for i in range(n):

        # display original

        ax = plt.subplot(2, n, i+1)

        plt.imshow(orig_imgs[300*i].reshape(img_width, img_height, channels))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)



        # display reconstruction

        ax = plt.subplot(2, n, i +1 + n)

        plt.imshow(decoded_imgs[300*i].reshape(img_width, img_height, channels))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.show()

    

def train_model(model, X_train, y_train, X_valid, y_valid, data_aug = False, epochs=50, batch_size=512):

    er = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

#     cp = ModelCheckpoint(filepath = 'best_model.h5',save_best_only = True,verbose=1)

    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=0.0001)

    callbacks = [er, lr]

    

    if not data_aug:  

        print('Training without data augmentation...')

        history = model.fit(X_train, y_train, batch_size=batch_size,

                                 epochs=epochs,

                                 verbose=1, callbacks=callbacks,

                                 validation_data=(X_valid,y_valid))

        return history

    else:

        print('Training with data augmentation...')

        train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        train_set_ae = train_datagen.flow(X_train, y_train, batch_size=batch_size)



        validation_datagen = ImageDataGenerator()

        validation_set_ae = validation_datagen.flow(X_valid, y_valid, batch_size=batch_size)

        

        history = model.fit_generator(train_set_ae,

                                           epochs=epochs,

                                           steps_per_epoch=np.ceil(X_train.shape[0]/batch_size),

                                           verbose=1, callbacks=callbacks,

                                           validation_data=(validation_set_ae),

                                           validation_steps=np.ceil(X_valid.shape[0]/batch_size))

        

        return history

    

def plot_loss_accuracy(history, plot_loss_only= False):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    ax1.plot(history.history['loss'])

    ax1.plot(history.history['val_loss'])

    ax1.set_title('Model Loss')

    ax1.set_ylabel('Loss')

    ax1.set_xlabel('Epoch')

    ax1.legend(['Train', 'Val'], loc='lower right')    

        

    if not plot_loss_only:

        ax2.plot(history.history['acc'])

        ax2.plot(history.history['val_acc'])

        ax2.set_title('Model accuracy')

        ax2.set_ylabel('Accuracy')

        ax2.set_xlabel('Epoch')

        ax2.legend(['Train', 'Val'], loc='lower right')    

    plt.show()
no_of_classes = 10

dict = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}

img_width, img_height = 32, 32

epochs = 50

batch_size = 512



if K.image_data_format()=='channels_first':

    input_shape=(3, img_width, img_height)

else:

    input_shape=(img_width,img_height,3)
def preprocess_data(X_train_images, X_test_images, y_train_labels, y_test_labels, no_of_classes):

    #normalize

    X_train_images = X_train_images.astype('float32')/255.

    X_test = X_test_images.astype('float32')/255.

    

    #split training data further into training and validation set

    X_train, X_valid, y_train_labels, y_valid_labels = train_test_split(X_train_images, y_train_labels, test_size=0.2, random_state=42, shuffle= True)

    print('X_train.shape: ', X_train.shape)

    print('X_valid.shape: ', X_valid.shape)

    print('y_train_labels.shape: ', y_train_labels.shape)

    print('y_valid_labels.shape: ', y_valid_labels.shape)

    

    #The target variable is converted to one-hot encoded data using the utils.to_categorical function of the keras library.

    y_train = to_categorical(y_train_labels, no_of_classes)

    y_valid = to_categorical(y_valid_labels, no_of_classes)

    y_test = to_categorical(y_test_labels, no_of_classes)

    print('y_train.shape: ', y_train.shape)

    print('y_test.shape: ', y_test.shape)

    

    return ((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = preprocess_data(X_train_images, X_test_images, y_train_labels, y_test_labels, no_of_classes)
def create_simple_conv_model():

    model = Sequential()

    model.add(Conv2D(64,(3,3),input_shape=input_shape, padding="same"))

    model.add(Activation('relu'))

    model.add(BatchNormalization())    

    model.add(MaxPool2D(pool_size=(2,2)))    

    

    model.add(Conv2D(128,(3,3),input_shape=input_shape, padding="same"))

    model.add(Activation('relu'))

    model.add(BatchNormalization())    

    model.add(MaxPool2D(pool_size=(2,2)))

    

    model.add(Flatten())

    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(no_of_classes))

    model.add(Activation('softmax'))

        

    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['acc'])

    

    return model



conv_model = create_simple_conv_model()

conv_model.summary()
history = train_model(conv_model, X_train, y_train, X_valid, y_valid, data_aug = True, epochs=epochs, batch_size=batch_size)
plot_loss_accuracy(history)
print('Test accuracy for benchmark model= {}'.format(conv_model.evaluate(X_test, y_test)[1]))
predictions = conv_model.predict(X_test)

report(predictions, y_test)
def create_block(input, chs): ## Convolution block of 2 layers

    x = input

    for i in range(2):

        x = Conv2D(chs, 3, padding="same")(x)

        x = Activation("relu")(x)

        x = BatchNormalization()(x)

    return x



def loss_function(y_true, y_pred):  ## loss function for using in autoencoder models

    mses = mean_squared_error(y_true, y_pred)

    return K.sum(mses, axis=(1,2))
def create_convolution_ae():

    input_img = Input((32,32,3))

    

    #encoder

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)

    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    x = BatchNormalization()(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)    



    # at this point the representation is ...



    #decoder

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)

    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)



    autoencoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

    encoder = Model(input_img, encoded)

    return encoder, autoencoder
# def create_convolution_ae2():

#     input = Input((32,32,3))

    

#     # Encoder

#     block1 = create_block(input, 32)

#     x = MaxPool2D(2)(block1)

#     block2 = create_block(x, 64)

#     x = MaxPool2D(2)(block2)

    

#     #Middle

#     middle = create_block(x, 128)

    

#     # Decoder

#     up1 = UpSampling2D((2,2))(middle)

#     block3 = create_block(up1, 64)

#     #up1 = UpSampling2D((2,2))(block3)

#     up2 = UpSampling2D((2,2))(block3)

#     block4 = create_block(up2, 32)

#     #up2 = UpSampling2D((2,2))(block4)

    

#     # output

#     x = Conv2D(3, 1)(up2)

#     output = Activation("sigmoid")(x)

#     return Model(input, middle), Model(input, output)



# encoder_conv, autoencoder_conv = create_convolution_ae2()

# # autoencoder.compile(loss='categorical_crossentropy', optimizer=Adadelta())  # model_ae.compile(SGD(1e-3, 0.9), loss=loss_function) 

# autoencoder_conv.summary()
encoder_conv, autoencoder_conv = create_convolution_ae()

# autoencoder.compile(loss='categorical_crossentropy', optimizer=Adadelta())  # model_ae.compile(SGD(1e-3, 0.9), loss=loss_function) 

autoencoder_conv.summary()
#Train the autoencoder convolution model. No need to use labels as it is unsupervised learning

history = train_model(autoencoder_conv, X_train, X_train, X_valid, X_valid, data_aug = False, epochs=epochs, batch_size=batch_size)

print(history)
plot_loss_accuracy(history, plot_loss_only=True)
decoded_test_imgs_conv_ae = autoencoder_conv.predict(X_test)

decoded_valid_imgs_conv_ae = autoencoder_conv.predict(X_valid)
showOrigAndReconstructedImages(X_valid, decoded_valid_imgs_conv_ae, 32, 32, 3)
showOrigAndReconstructedImages(X_test, decoded_test_imgs_conv_ae, 32, 32, 3)
def classifier_dense(inp):

    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))

    #x = MaxPool2D()(input)

    x = Flatten()(input)

    #x = BatchNormalization()(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.64)(x)

    x = Dense(50, activation='relu')(x)

    #x = Reshape((-1, 1))(x)

    #x = Conv1D(128, (3,), activation='relu', padding='same')(x)

    #x = MaxPool1D()(x)

    #x = CuDNNLSTM(64)(x)

    #x = Flatten()(x)

    x = Dropout(0.4)(x)

    output = Dense(no_of_classes, activation='softmax')(x)

    return Model(input, output)



def classifier_conv(inp):

    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))

    x = Conv2D(1024, 3, padding="same")(input)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = MaxPool2D(2)(x)

    x = Dropout(0.5)(x)

    

    x = Conv2D(128, 3, padding="same")(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = MaxPool2D(2)(x)

    x = Dropout(0.5)(x)

    

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.35)(x)

    x = Dense(100, activation='relu')(x)

    x = Dropout(0.69)(x)

    

    output = Dense(no_of_classes, activation='softmax')(x)

    return Model(input, output)



def create_classifier(m, inp):  ## function for choosing dense/convolutional classifier model

    if m=='dense':

        classifier = classifier_dense(inp)

    elif m=='conv':

        classifier = classifier_conv(inp)    

    classifier.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['acc'])

    return classifier
#Extracting bottleneck features to use as inputs in the classifier model:

gist_train_ae = encoder_conv.predict(X_train)

gist_valid_ae = encoder_conv.predict(X_valid)

gist_test_ae = encoder_conv.predict(X_test)
decoder_ae_dense = create_classifier('dense', gist_train_ae)

decoder_ae_dense.summary()
history = train_model(decoder_ae_dense, gist_train_ae, y_train, gist_valid_ae, y_valid, data_aug = False, epochs=epochs, batch_size=batch_size)
plot_loss_accuracy(history)
print('Test accuracy for AE_dense model= {}'.format(decoder_ae_dense.evaluate(gist_test_ae, y_test)[1]))
show_test(decoder_ae_dense, X_test, y_test, gist_test_ae)
predictions = decoder_ae_dense.predict(gist_test_ae)

report(predictions, y_test)
decoder_ae_conv = create_classifier('conv', gist_train_ae)

decoder_ae_conv.summary()
hist = train_model(decoder_ae_conv, gist_train_ae, y_train, gist_valid_ae, y_valid, data_aug = False, epochs=epochs, batch_size=batch_size)
plot_loss_accuracy(history)
print('Test accuracy for AE_conv model= {}'.format(decoder_ae_conv.evaluate(gist_test_ae, y_test)[1]))
show_test(decoder_ae_conv, X_test, y_test, gist_test_ae)
predictions = decoder_ae_conv.predict(gist_test_ae)

report(predictions, y_test)