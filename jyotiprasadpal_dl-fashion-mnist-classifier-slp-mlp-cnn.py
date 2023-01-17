import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import keras

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



from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
def load_data(keras_datasets, first_layer="dense", channels=1, plot_images=False, class_names=[]):

    (x_train, y_train), (x_test, y_test) = keras_datasets.load_data()

    print('Before reshape - X_train.shape:', x_train.shape)

    print('Before reshape - X_test.shape:', x_test.shape)

    height=x_train.shape[1]

    width=x_train.shape[2]        

    # flatten 28*28 images to a 784 vector for each image

    num_pixels = height * width

    

    #plot images

    if plot_images:

        plot_images_with_labels(x_train, y_train, height, width, class_names, 25)

    

    if first_layer == "dense":

        # convert shape of x_train from (60000, 28, 28) to (60000, 784) - 784 columns per row        

        X_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')

        X_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')        

    elif first_layer == "conv2d":

        # Select class 6 images (class 6)

#         x_train = x_train[y_train.flatten() == 6]

        X_train = x_train.reshape((x_train.shape[0], height, width, channels)).astype('float32')

        X_test = x_test.reshape((x_test.shape[0], height, width, channels)).astype('float32')        

        print((x_train.shape[0],) + (height, width, channels))        

        

    print('After reshape - X_train.shape:', X_train.shape)

    print('After reshape - X_test.shape:', X_test.shape)

    print('Before rescaling:', X_train[0])

    #normalize the values between 0 and 1

    X_train = (X_train.astype(np.float32))/255

    X_test = (X_test.astype(np.float32))/255

    print('After rescaling:', X_train[0])

              

    #convert labels to categorical/dummy encoding so that we can use simple "categorical_crossentropy" as loss.

    print('Class label of first image before converting to categorical:', y_train[0])

    # one hot encode outputs

    y_train = np_utils.to_categorical(y_train)

    y_test = np_utils.to_categorical(y_test)

    num_classes = y_test.shape[1]

    print('Total number of classes:', num_classes)

    print('Class label of first image after converting to categorical:', y_train[0])

              

    return (X_train, y_train, X_test, y_test, height, width)



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
NUM_CLASSES=10

CLASS_NAMES = [

    "T-shirt/top",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandal",

    "Shirt",

    "Sneaker",

    "Bag",

    "Ankle boot"

]

EPOCHS=50

BATCH_SIZE=1000

CHANNELS=1

VERBOSE=1

METRICS=['acc']
X_train, y_train, X_test, y_test, IMG_HEIGHT, IMG_WIDTH = load_data(keras.datasets.fashion_mnist, first_layer="dense", channels=1, plot_images=True, class_names=CLASS_NAMES)
X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn, IMG_HEIGHT, IMG_WIDTH = load_data(keras.datasets.fashion_mnist, first_layer="conv2d", channels=1, plot_images=False)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

X_train_cnn, X_valid_cnn, y_train_cnn, y_valid_cnn = train_test_split(X_train_cnn, y_train_cnn, test_size=0.20, random_state=42)
#create single layer network called perceptron

def build_slp_model(height, width, nb_classes, metrics):

    model = keras.models.Sequential()

#     model.add(keras.layers.Flatten(input_shape=(height, width)))

    model.add(keras.layers.Dense(nb_classes, input_dim=(height* width), use_bias=False, activation='softmax')) #no hidden layers, all inputs connected to all outputs

    model.compile(loss = 'categorical_crossentropy',

              optimizer=keras.optimizers.Adam(lr=.0001),#optimizer='adam',              

              metrics=metrics)

    return model



model_slp = build_slp_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, METRICS)

model_slp.summary()
history_slp = train_model(model_slp, X_train, y_train, X_valid=X_valid, y_valid=y_valid, data_aug = False, 

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
#predictions on unseen test data

X_new = X_test[:3]

y_proba = model_slp.predict(X_new)

print(y_proba.round(2))



y_pred = model_slp.predict_classes(X_new)

print(y_pred)



print(np.array(CLASS_NAMES)[y_pred])
def build_mlp_model(height, width, nb_classes, metrics):

    #create the multilayer preceptron model with 4 hidden layers

    model = keras.models.Sequential()

#     model.add(keras.layers.Flatten(input_shape=(height, width)))

    model.add(keras.layers.Dense(256, input_dim=(height* width), activation='relu'))

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.Dense(nb_classes, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy',

              optimizer='rmsprop',                           #optimizer=keras.optimizers.SGD(lr=.001),#optimizer='sgd',              

              metrics=metrics)

    return model



model_mlp = build_mlp_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, METRICS)

model_mlp.summary()
model_mlp.layers
hidden1 = model_mlp.layers[1]

weights, biases = hidden1.get_weights()

print(weights.shape)

print(biases.shape)
history_mlp = train_model(model_mlp, X_train, y_train, X_valid=X_valid, y_valid=y_valid, data_aug = False, 

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
#predictions on unseen test data

X_new = X_test[:3]

y_proba = model_mlp.predict(X_new)

print(y_proba.round(2))



y_pred = model_mlp.predict_classes(X_new)

print(y_pred)



print(np.array(CLASS_NAMES)[y_pred])
def create_simple_conv_model(image_height=IMG_HEIGHT, image_width=IMG_WIDTH, channels=CHANNELS, nb_classes=NUM_CLASSES, metrics=METRICS, optimizer='adam'):    

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

        

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    return model



model_cnn = create_simple_conv_model(IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES, METRICS)

model_cnn.summary()
history_cnn = train_model(model_cnn, X_train_cnn, y_train_cnn, X_valid=X_valid_cnn, y_valid=y_valid_cnn, data_aug = False, 

            best_model_name='best_model_cnn.h5', epochs=EPOCHS, batch_size=BATCH_SIZE,verbose=VERBOSE)
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
#predictions on unseen test data

X_new = X_test_cnn[:3]

y_proba = model_cnn.predict(X_new)

print(y_proba.round(2))



y_pred = model_cnn.predict_classes(X_new)

print(y_pred)



print(np.array(CLASS_NAMES)[y_pred])
# create model

model = KerasClassifier(build_fn=create_simple_conv_model, verbose=0)



# grid search epochs, batch size and optimizer

optimizers = ['adam', 'rmsprop', 'SGD']

# init = ['glorot_uniform', 'normal', 'uniform']

epochs = [10, 30, 50]

batches = [500, 1000, 5000]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)

grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result = grid.fit(X_train_cnn, y_train_cnn)



# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))