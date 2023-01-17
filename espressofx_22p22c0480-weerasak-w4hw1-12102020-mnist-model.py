import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

import itertools



import tensorflow

from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split



train = pd.read_csv("../input/dataset-1024/dataset_1024.csv", index_col=None)

X_train, X_test, y_train, y_test = train_test_split(train.drop("label", axis=1),train.label, test_size=0.33)
#from sklearn.utils import class_weight



#class_weights = class_weight.compute_class_weight('balanced',

#                                                  np.unique(y_train), 

#                                                  y_train)

#class_weights = dict(enumerate(class_weights))

#class_weights
IMG_SIZE=32

BATCH_SIZE = 32
g = sns.countplot(y_train)

y_train.value_counts()
# Normalize the data

X_train = X_train/255.0

X_test = X_test/255.0
# Reshape image in 3 dimensions (height = 32px, width = 32px , canal = 1)

X_train = X_train.values.reshape(-1,IMG_SIZE,IMG_SIZE,1)

X_test = X_test.values.reshape(-1,IMG_SIZE,IMG_SIZE,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes = 10)
# Split the train and the validation set for the fitting

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
# Some examples

plt.imshow(X_train[1][:,:,0],cmap='gray')
def LeNet5(input_shape = (32, 32, 1), classes = 10):

    model = Sequential([

        Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu', input_shape = input_shape),

        MaxPooling2D(pool_size = 2, strides = 2),



        Conv2D(filters = 16, kernel_size = 5, strides = 1, activation = 'relu'),

        MaxPooling2D(pool_size = 2, strides = 2),



        Flatten(),

        Dense(120, activation = 'relu'),

        Dense(84, activation = 'relu'),

        

        Dense(classes, activation = 'softmax')

    ])

    return model
def LeNet5v1(input_shape = (32, 32, 1), classes = 10):

    model = Sequential([

        Conv2D(6, kernel_size = 5, strides = 1, activation = 'relu', input_shape = input_shape),   

        

        Conv2D(6, kernel_size = 5, strides = 1, activation = 'relu'),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.25),

        

        Conv2D(16, kernel_size = 5, strides = 1, activation = 'relu', kernel_regularizer = l2(0.01)),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.25),

        

        Flatten(),

        Dense(120, activation = 'relu'),

        Dense(84, activation = 'relu'),

        Dense(classes, activation = 'softmax')  

    ])

    return model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization



def LeNet5v2(input_shape = (32, 32, 1), classes = 10):

    model = Sequential([

        Conv2D(32, kernel_size = 5, strides = 1, activation = 'relu', input_shape = input_shape, kernel_regularizer=l2(0.0005)),

        Conv2D(32, kernel_size = 5, strides = 1, use_bias=False),  

        BatchNormalization(),



        Activation("relu"),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.25),



        Conv2D(64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=l2(0.0005)),

        Conv2D(64, kernel_size = 3, strides = 1, use_bias=False),

        BatchNormalization(),



        Activation("relu"),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.25),

        Flatten(),



        Dense(256,use_bias=False),

        BatchNormalization(),

        Activation("relu"),



        Dense(128, use_bias=False),

        BatchNormalization(), 

        Activation("relu"),





        Dense(84, use_bias=False),

        BatchNormalization(), 

        Activation("relu"),

        

        Dropout(0.25),

        Dense(classes, activation = 'softmax')

    ])

    return model
def MyNet(input_shape = (32, 32, 1), classes = 10):

    model = Sequential([

        Conv2D(32, kernel_size = 5,padding = 'Same', activation ='relu', input_shape = input_shape),

        Conv2D(32, kernel_size = 5,padding = 'Same', activation ='relu'),

        MaxPool2D(pool_size=2),

        Dropout(0.25),



        Conv2D(64, kernel_size = 5,padding = 'Same', activation ='relu'),

        Conv2D(64, kernel_size = 5,padding = 'Same', activation ='relu'),

        MaxPool2D(pool_size=2, strides=2),

        Dropout(0.25),



        Flatten(),

        Dense(256, activation = "relu"),

        Dropout(0.5),

        Dense(classes, activation = "softmax")

    ])

    return model
def MyNetV1(input_shape = (32, 32, 1), classes = 10):

    model = Sequential([

        Conv2D(32, kernel_size = 5,padding = 'Same', activation ='relu', input_shape = input_shape),

        Conv2D(32, kernel_size = 5,padding = 'Same', activation ='relu'),

        

        BatchNormalization(),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.25),



        Conv2D(64, kernel_size = 5,padding = 'Same', activation ='relu'),

        Conv2D(64, kernel_size = 5,padding = 'Same', activation ='relu'),

        

        BatchNormalization(),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.25),



        Flatten(),

        Dense(256, activation = "relu"),

        BatchNormalization(), 

        Activation("relu"),

        Dropout(0.5),

        

        Dense(classes, activation = "softmax")

    ])

    return model
def MyNetV2(input_shape = (32, 32, 1), classes = 10):

    model = Sequential([

        Conv2D(32, kernel_size = 5,padding = 'Same', activation ='relu', input_shape = input_shape),

        Conv2D(32, kernel_size = 5,padding = 'Same', activation ='relu'),

        

        BatchNormalization(),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.5),



        Conv2D(64, kernel_size = 5,padding = 'Same', activation ='relu'),

        Conv2D(64, kernel_size = 5,padding = 'Same', activation ='relu'),

        

        BatchNormalization(),

        MaxPooling2D(pool_size = 2, strides = 2),

        Dropout(0.5),



        Flatten(),



        Dense(256),

        BatchNormalization(),

        Activation("relu"),



        Dropout(0.25),

        

        Dense(classes, activation = "softmax")

    ])

    return model
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)

        #zoom_range = 0.05, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
def compile_model(model):

    # Set a learning rate annealer

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)



    checkpoint_filepath = 'th_mnist_model_20201015.h5'

    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(

        filepath=checkpoint_filepath,

        save_weights_only=False,

        monitor='val_accuracy',

        mode='max',

        save_best_only=True,

        verbose=1)

    callbacks=[learning_rate_reduction,model_checkpoint_callback]



    # Define the optimizer

    optimizer = RMSprop(lr=1e-2, rho=0.9, epsilon=1e-08, decay=0.0)

    #optimizer = Adam(lr=0.0001, epsilon=1e-08, decay=0.0) #0.96



    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    return model,callbacks
model,callbacks = compile_model(MyNetV1(input_shape = (IMG_SIZE, IMG_SIZE, 1), classes = 10))

#model.summary()
EPOCHES = 50

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              epochs = EPOCHES, validation_data = (X_val,y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // BATCH_SIZE, 

                              callbacks=callbacks)

                              #class_weight=class_weights)
def plot_acc(history):

    # Plot the loss and accuracy curves for training and validation 

    fig, ax = plt.subplots(2,1, sharex=True, figsize=(20,10))

    ax[0].plot(history.history['loss'], color='b', label="Training loss")

    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)

    ax[0].set_ylim([0.0, 0.5])

    ax[0].set_xlim([0, EPOCHES-1])



    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

    legend = ax[1].legend(loc='best', shadow=True)

    ax[1].set_ylim([0.90, 1.0])

    ax[1].set_xlim([0, EPOCHES-1])
def plot_confusion_matrix(cm, classes,

                          normalize=False,

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



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
plot_acc(history)
fig, ax = plt.subplots(figsize=(8, 8))



y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
print(classification_report(y_true, y_pred_classes))
tensorflow.__version__