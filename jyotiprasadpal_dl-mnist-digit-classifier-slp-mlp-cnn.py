import numpy as np

import matplotlib.pyplot as plt



from keras.datasets import mnist

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential, Model, load_model

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, Convolution2D, MaxPool2D, MaxPooling2D, ZeroPadding2D, BatchNormalization

from numpy.random import permutation

from keras import optimizers

from keras.optimizers import SGD, Adam

from keras import backend as K

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.utils import to_categorical, np_utils



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, plot_confusion_matrix, roc_auc_score, roc_curve
def plot_images_with_labels(X, y, img_height, img_width, class_names, nb_count=25):

    plt.figure(figsize=(10, 10))

    for i in range(nb_count):

        plt.subplot(np.sqrt(nb_count), np.sqrt(nb_count), i + 1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(X[i].reshape((img_height,img_width)), cmap=plt.get_cmap('gray'))

        label_index = int(y[i])

        plt.title(class_names[label_index])

    plt.show()



def train_model(model, X_train, y_train, X_valid=None, y_valid=None, validation_split=0.20, data_aug = False, best_model_name='best_model.h5', epochs=50, batch_size=512,verbose=1):

    er = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=verbose)

    cp = ModelCheckpoint(filepath = best_model_name, save_best_only = True,verbose=verbose)

#     lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=0.0001)

    callbacks = [cp, er]

    

    if not data_aug and X_valid is not None:  

        print('Training without data augmentation...')

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=verbose, callbacks=callbacks, validation_data=(X_valid,y_valid))

        return history

    elif not data_aug and X_valid is None:

        print('Training without data augmentation...')

        history = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs, verbose=verbose, shuffle=True, callbacks=callbacks, validation_split=validation_split)

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

                                           verbose=verbose, callbacks=callbacks,

                                           validation_data=(validation_set_ae),

                                           validation_steps=np.ceil(X_valid.shape[0]/batch_size))

        

        return history

    

def plot_loss_and_metrics(history, plot_loss_only= False, metrics=['acc']):

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics)+1, figsize=(20, 4))

    axes[0].plot(history.history['loss'])

    axes[0].plot(history.history['val_loss'])

    axes[0].set_title('Model Loss')

    axes[0].set_ylabel('Loss')

    axes[0].set_xlabel('Epoch')

    axes[0].legend(['Train', 'Val'], loc='lower right')    

        

    if not plot_loss_only:

        axes[1].plot(history.history['acc'])

        axes[1].plot(history.history['val_acc'])

        axes[1].set_title('Model Accuracy')

        axes[1].set_ylabel('Accuracy')

        axes[1].set_xlabel('Epoch')

        axes[1].legend(['Train', 'Val'], loc='lower right')  

        

        if 'mae' in metrics:

            axes[2].plot(history.history['mae'])

            axes[2].plot(history.history['val_mae'])

            axes[2].set_title('Model Mean Absolute Error')

            axes[2].set_ylabel('Mean Absolute Error')

            axes[2].set_xlabel('Epoch')

            axes[2].legend(['Train', 'Val'], loc='lower right') 

        if 'mse' in metrics:

            axes[3].plot(history.history['mse'])

            axes[3].plot(history.history['val_mse'])

            axes[3].set_title('Model Mean Squared Error')

            axes[3].set_ylabel('Mean Squared Error')

            axes[3].set_xlabel('Epoch')

            axes[3].legend(['Train', 'Val'], loc='lower right')

            

    plt.show()

    

def plot_roc_curve(fpr,tpr): 

  import matplotlib.pyplot as plt

  plt.plot(fpr,tpr) 

  plt.axis([0,1,0,1]) 

  plt.xlabel('False Positive Rate') 

  plt.ylabel('True Positive Rate') 

  plt.show()  

    

def load_evaluate_predict(fileName, X_test, y_test, nb_round=0, print_first=1, metrics=['acc']):

    #load best model, evaluate and predict on unseen data    

    best_model = load_model(fileName)

    results = best_model.evaluate(X_test, y_test)    

    print('Test loss = {}'.format(np.round(results[0], 2)))

    print('Test accuracy = {}'.format(np.round(results[1], 2)))

    if len(metrics)>1:

        print('Test ' + metrics[1] + '= {}'.format(np.round(results[2], 2)))

        print('Test ' + metrics[2] + '= {}'.format(np.round(results[3], 2)))



    y_pred_proba = best_model.predict(X_test)

    for i in range(print_first):

        print('')

        print("   Actual:", y_test[i])

        print('Predicted:', np.round(y_pred_proba[i], nb_round))

    

    return best_model, y_pred_proba



def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import matplotlib.pyplot as plt

    import itertools

    

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=20)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)

    plt.yticks(tick_marks, classes, fontsize=12)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label', fontsize=12)

    plt.xlabel('Predicted label', fontsize=12)

    

def report_metrics(y_test, y_pred, y_pred_proba, classes, multiclass=False):

    #confusion matrix

    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))

    plot_confusion_matrix(cnf_matrix, classes=classes, title="Confusion matrix")

    plt.show()



    #classification report

    print('classification report:\n', classification_report(y_test, y_pred))

    

    if not multiclass:

        #calculate the roc auc score

        auc = roc_auc_score(y_test, y_pred_proba)

        print('AUC: %.3f' % auc)

    

        #plot the roc curve

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_proba)

        print('ROC curve:\n')

        plot_roc_curve(fpr_keras, tpr_keras)
# load the MNIST dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('Before reshape - Original X_train.shape:', X_train.shape)
# plot first 25 images as gray scale

NUM_CLASSES=10

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

EPOCHS=50

BATCH_SIZE=5000

IMG_HEIGHT= X_train.shape[1]

IMG_WIDTH = X_train.shape[2]

CHANNELS=1

VERBOSE=1

METRICS=['acc']
plot_images_with_labels(X_train, y_train, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES, 25)
# flatten 28*28 images to a 784 vector for each image

num_pixels = IMG_HEIGHT * IMG_WIDTH



#for inpput to Conv2D layer

X_train_cnn = X_train.reshape((X_train.shape[0], IMG_HEIGHT, IMG_WIDTH, CHANNELS)).astype('float32')

X_test_cnn = X_test.reshape((X_test.shape[0], IMG_HEIGHT, IMG_WIDTH, CHANNELS)).astype('float32')

print('After reshape - Original X_train_cnn.shape:', X_train_cnn.shape)



#for input to dense layer

X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')

X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

print('After reshape - Original X_train.shape:', X_train.shape)

print(X_train[0])
# normalize inputs from 0-255 to 0-1

#for cnn layer

X_train_cnn = X_train_cnn / 255.

X_test_cnn = X_test_cnn / 255.

print('After reshape - Scaled X_train_cnn.shape:', X_train_cnn.shape)



#for dense layer

X_train = X_train / 255.

X_test = X_test / 255.

print('After reshape - Scaled X_train.shape:', X_train.shape)

print(X_train[0])
print('Class label of first image before converting to categorical:', y_train[0])

# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

print('Total number of classes:', num_classes)
print('Class label of first image after converting to categorical:', y_train[0])
def build_SLP_model(metrics):

    model = Sequential()

    model.add(Dense(1024, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    #Compile model

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=metrics)

    return model



model_slp = build_SLP_model(METRICS)

model_slp.summary()
history_slp = train_model(model_slp, X_train, y_train, X_valid=None, y_valid=None, validation_split=0.20, data_aug = False, 

            best_model_name='best_model_slp.h5', epochs=EPOCHS, batch_size=BATCH_SIZE,verbose=VERBOSE)
# print the loss and accuracy

plot_loss_and_metrics(history_slp)

_, y_pred_proba_slp = load_evaluate_predict('best_model_slp.h5', X_test, y_test, nb_round=0, print_first=1, metrics=METRICS)



print(y_pred_proba_slp[0])

print(y_pred_proba_slp[0].shape)

## Get most likely class

y_pred_slp = np.argmax(y_pred_proba_slp, axis=1)

print(y_pred_slp)



#Confusion Matrix, Classification report, ROC curve

report_metrics(np.argmax(y_test, axis=1), y_pred_slp, y_pred_proba_slp, CLASS_NAMES, multiclass=True)

def basic_MLP_model(metrics):

    model = Sequential()

    model.add(Dense(1024, activation='relu', input_shape=(784,)))

    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model

    

model_mlp = basic_MLP_model(METRICS)

model_mlp.summary()
history_mlp = train_model(model_mlp, X_train, y_train, X_valid=None, y_valid=None, validation_split=0.20, data_aug = False, 

            best_model_name='best_model_mlp.h5', epochs=EPOCHS, batch_size=BATCH_SIZE,verbose=VERBOSE)
# print the loss and accuracy

plot_loss_and_metrics(history_mlp)

_, y_pred_proba_mlp = load_evaluate_predict('best_model_mlp.h5', X_test, y_test, nb_round=0, print_first=1, metrics=METRICS)



print(y_pred_proba_mlp[0])

print(y_pred_proba_mlp[0].shape)

## Get most likely class

y_pred_mlp = np.argmax(y_pred_proba_mlp, axis=1)

print(y_pred_mlp)



#Confusion Matrix, Classification report, ROC curve

report_metrics(np.argmax(y_test, axis=1), y_pred_mlp, y_pred_proba_mlp, CLASS_NAMES, multiclass=True)
def create_simple_conv_model(image_height, image_width, channels, nb_classes, metrics):    

    # number of convolutional filters to use

    nb_filters = 32   

    # convolution kernel size

    nb_conv = 3

     # size of pooling area for max pooling

    nb_pool = 2

    model = Sequential()

    model.add(Conv2D(filters=nb_filters, kernel_size=(nb_conv,nb_conv), strides=(1, 1), activation='relu', input_shape=(image_height, image_width, channels)))  

    model.add(BatchNormalization())    

    model.add(MaxPool2D(pool_size=(nb_pool,nb_pool)))   

#     model.add(Dropout(0.5))   



    model.add(Flatten())

    model.add(Dense(128, activation='relu'))    

#     model.add(Dropout(0.5))

    

    model.add(Dense(nb_classes, activation='softmax'))   

        

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=metrics)

    return model



model_cnn = create_simple_conv_model(IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES, METRICS)

model_cnn.summary()
history_cnn = train_model(model_cnn, X_train_cnn, y_train, X_valid=None, y_valid=None, validation_split=0.20, data_aug = False, 

            best_model_name='best_model_cnn.h5', epochs=EPOCHS, batch_size=100,verbose=VERBOSE)
# print the loss and accuracy

plot_loss_and_metrics(history_cnn)

_, y_pred_proba_cnn=  load_evaluate_predict('best_model_cnn.h5', X_test_cnn, y_test, nb_round=0, print_first=1, metrics=METRICS)



print(y_pred_proba_cnn[0])

print(y_pred_proba_cnn[0].shape)

## Get most likely class

y_pred_cnn = np.argmax(y_pred_proba_cnn, axis=1)

print(y_pred_cnn)



#Confusion Matrix, Classification report, ROC curve

report_metrics(np.argmax(y_test, axis=1), y_pred_cnn, y_pred_proba_cnn, CLASS_NAMES, multiclass=True)