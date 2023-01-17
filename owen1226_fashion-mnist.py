# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline

import math
import time

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Add, Multiply, Average, Maximum, Dense, Activation, ZeroPadding2D, BatchNormalization, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform, he_uniform
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, layer_utils, plot_model
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
# Get train and test data
data_train = pd.read_csv('../input/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashion-mnist_test.csv')
# Preprocess
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
IMG_SHAPE = (28, 28, 1)   # rows, cols, channels
# Train dataset
X_train = np.array(data_train.iloc[:, 1:])
X_train = X_train.reshape(X_train.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
X_train = X_train.astype('float32')  # string to float
X_train /= 255
Y_train = to_categorical(np.array(data_train.iloc[:, 0]))  # CLASSES to one_hot
# Test dataset
X_test_orig = np.array(data_test.iloc[:, 1:])
X_test = X_test_orig.reshape(X_test_orig.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
X_test = X_test.astype('float32')
X_test /= 255
Y_test = to_categorical(np.array(data_test.iloc[:, 0]))
def plot_image(image, label):
    if not np.isscalar(label):
        label = np.argmax(label)
        
    # plt.figure(figsize=(10,5))
    plt.imshow(np.squeeze(image.reshape(IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])), interpolation='nearest')
    plt.title(CLASSES[int(label)])



image_id = 0
plot_image(X_test[image_id, :], Y_test[image_id])
def MyModel(input_shape, num_classes=2):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides = (1, 1), padding='same', kernel_initializer=glorot_uniform(), name = 'Conv1')(X)
    X = BatchNormalization(axis = 3, name = 'BN1')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Conv2D(64, (3, 3), strides = (1, 1), padding='same', kernel_initializer=glorot_uniform(), name = 'Conv2')(X)
    X = BatchNormalization(axis = 3, name = 'BN2')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding='same', kernel_initializer=glorot_uniform(), name = 'Conv3')(X)
    X = BatchNormalization(axis = 3, name = 'BN3')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='MP')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    
    # FULLY CONNECTED
    X = Dense(128, activation='relu', name='FC1')(X)
    
    if num_classes > 2:
        X = Dense(num_classes, activation='softmax', name='FC2')(X)
    else:
        X = Dense(1, activation='sigmoid', name='FC2')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='CNN')

    return model
model = MyModel(IMG_SHAPE, num_classes=len(CLASSES))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0003, decay=1e-6, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only=True)   # Save the best model
hist = model.fit(X_train, Y_train, batch_size=128, callbacks=[monitor, checkpoint], epochs=30, shuffle=True, verbose=1, validation_split=0.01)
def plot_train_history(history):
    # plot the cost and accuracy 
    loss_list = history['loss']
    val_loss_list = history['val_loss']
    accuracy_list = history['acc']
    val_accuracy_list = history['val_acc']
    # epochs = range(len(loss_list))

    # plot the cost
    plt.plot(loss_list, 'b', label='Training cost')
    plt.plot(val_loss_list, 'r', label='Validation cost')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Training and validation cost')
    plt.legend()
    
    plt.figure()
    
    # plot the accuracy
    plt.plot(accuracy_list, 'b', label='Training accuracy')
    plt.plot(val_accuracy_list, 'r', label='Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title('Training and validation accuracy')
    plt.legend()



plot_train_history(hist.history)
score = model.evaluate(X_test, Y_test)

print ("Test Loss = " + str(score[0]))
print ("Test Accuracy = " + str(score[1]))
Y_test_pred = model.predict(X_test, verbose=2)
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, classification_report

def analyze(Y, Y_pred, classes, activation="softmax"):
    if activation == "sigmoid":
        Y_cls = Y
        Y_pred_cls = (Y_pred > 0.5).astype(float)
    elif activation == "softmax":
        Y_cls = np.argmax(Y, axis=1)
        Y_pred_cls = np.argmax(Y_pred, axis=1)
    
    
    accuracy = accuracy_score(Y_cls, Y_pred_cls)
    print("Accuracy score: {}\n".format(accuracy))
    
    
    rmse = np.sqrt(mean_squared_error(Y, Y_pred))
    print("RMSE score: {}\n".format(rmse))

    
    # plot Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(Y_cls, Y_pred_cls)
    print(cm)
    # Plot the confusion matrix as an image.
    plt.matshow(cm)
    # Make various adjustments to the plot.
    num_classes = len(classes)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    
    # plot Classification Report
    print("Classification Report:")
    print(classification_report(Y_cls, Y_pred_cls, target_names=classes))



analyze(Y_test, Y_test_pred, CLASSES, "softmax")
def plot_mislabeled(X, Y, Y_pred, classes, activation="softmax", num_images = 0):
    """
    Plots images where predictions and truth were different.
    
    X -- original image data - shape(m, img_rows*img_cols)
    Y -- true labels - eg. [2,3,4,3,1,1]
    Y_pred -- predictions - eg. [2,3,4,3,1,2]
    """
    
    num_col = 5
    
    if activation == "sigmoid":
        Y_cls = Y
        Y_pred_cls = (Y_pred > 0.5).astype(float)
    elif activation == "softmax":
        Y_cls = np.argmax(Y, axis=1)
        Y_pred_cls = np.argmax(Y_pred, axis=1)
    
    mislabeled_indices = np.where(Y_cls != Y_pred_cls)[0]
    
    if num_images < 1:
        num_images = len(mislabeled_indices)
    
    fig, axes = plt.subplots(math.ceil(num_images/num_col), num_col, figsize=(25,20))

    for i, index in enumerate(mislabeled_indices[:num_images]):
#         plt.subplot(2, num_images, i + 1)
#         plt.imshow(X[index, :].reshape(IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]), interpolation='nearest')
#         plt.axis('off')
#         plt.title("Prediction: " + classes[p[index]] + " \n Class: " + classes[int(y[index])])
        row, col = i//num_col, i%num_col
        img = np.squeeze(X[index, :].reshape(IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))

        axes[row, col].imshow(img, interpolation='nearest')
        axes[row, col].axis('off')
        axes[row, col].set_title("Id: {}\nPrediction: {} - {}\nClass: {}".format(index, classes[int(Y_pred_cls[index])], np.amax(Y_pred[index]), classes[int(Y_cls[index])]))



plot_mislabeled(X_test, Y_test, Y_test_pred, CLASSES, "softmax", 20)
def plot_conv_layers(image, model):
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0))

    images_per_row = 16
    
    for layer_name, layer_activation in zip(layer_names, activations):
        if layer_name.startswith('Conv'):
            _, height, width, num_filters = layer_activation.shape   # image height and width, and size of channel
            n_rows = num_filters // images_per_row
            display_grid = np.zeros((n_rows * height, images_per_row * width))

            for row in range(n_rows):
                for col in range(images_per_row):
                    channel_image = layer_activation[0, :, :, row * images_per_row + col]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                    display_grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = channel_image

            plt.figure(figsize=(images_per_row *2, n_rows *2))
            plt.title(layer_name)
            plt.grid(False)
            plt.axis('off')
#             plt.imshow(display_grid, aspect='auto', interpolation='nearest', cmap='binary')
            plt.imshow(display_grid, aspect='auto', interpolation='nearest', cmap='viridis')


        
image_id = 0
plot_image(X_test[image_id], Y_test[image_id])
plot_conv_layers(X_test[image_id], model)