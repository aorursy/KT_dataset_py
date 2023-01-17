import numpy as np

import pandas as pd



from keras.layers import Conv3D, MaxPool3D, Flatten, Dense

from keras.layers import Dropout, Input, BatchNormalization, Activation

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.losses import categorical_crossentropy

from keras.optimizers import adam, adadelta

from keras.models import Model

from keras.utils.np_utils import to_categorical



import matplotlib.pyplot as plt

from matplotlib.pyplot import cm

%matplotlib inline



import seaborn as sns



import h5py



import os

print(os.listdir("../input"))
with h5py.File('../input/3d-mnist/full_dataset_vectors.h5', 'r') as dataset:

    x_train, x_test = dataset["X_train"][:], dataset["X_test"][:]

    y_train, y_test = dataset["y_train"][:], dataset["y_test"][:]



print ("x_train shape: ", x_train.shape)

print ("y_train shape: ", y_train.shape)



print ("x_test shape:  ", x_test.shape)

print ("y_test shape:  ", y_test.shape)
y_train[1], y_test[1]
## Introduce the channel dimention in the input dataset 

xtrain = np.ndarray((x_train.shape[0], 4096, 3))

xtest = np.ndarray((x_test.shape[0], 4096, 3))



# Translate data to color

def add_rgb_dimention(array):

    scalar_map = cm.ScalarMappable(cmap="Oranges")

    return scalar_map.to_rgba(array)[:, : -1]



## iterate in train and test, add the rgb dimention 

for i in range(x_train.shape[0]):

    xtrain[i] = add_rgb_dimention(x_train[i])

for i in range(x_test.shape[0]):

    xtest[i] = add_rgb_dimention(x_test[i])



## convert to 1 + 4D space (1st argument represents number of rows in the dataset)

xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)

xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)



## convert target variable into one-hot

y_train = to_categorical(y_train, 10)
xtrain.shape, y_train.shape
## input layer

input_layer = Input((16, 16, 16, 3))



## convolutional layers

x = Conv3D(filters=8, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(input_layer)

x = BatchNormalization()(x)

x = Activation("relu")(x)

x = Conv3D(filters=16, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)



## Pooling layer

x = MaxPool3D(pool_size=(2, 2, 2))(x) # the pool_size (2, 2, 2) halves the size of its input



## convolutional layers

x = Conv3D(filters=32, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)

x = Conv3D(filters=64, kernel_size=(3, 3, 3), use_bias=False, padding='Same')(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)



## Pooling layer

x = MaxPool3D(pool_size=(2, 2, 2))(x)

x = Dropout(0.25)(x) #No more BatchNorm after this layer because we introduce Dropout



x = Flatten()(x)



## Dense layers

x = Dense(units=4096, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(units=1024, activation='relu')(x)

x = Dropout(0.5)(x)

output_layer = Dense(units=10, activation='softmax')(x)



## define the model with input layer and output layer

model = Model(inputs=input_layer, outputs=output_layer, name="3D-CNN")

model_name = model.name



#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

#"Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models which combines the best properties of the AdaGrad and RMSProp algorithms.

#It provides an optimization algorithm that can handle sparse gradients on noisy problems. The default configuration parameters do well on most problems.""

model.compile(loss='categorical_crossentropy', 

              optimizer=adam(), #default: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0

              metrics=['acc'])



model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



#patience:

patience_earlystop = 7

patience_ReduceLROnPlateau = 3



filepath = 'best_weight.h5'

mcp = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)

earlystop = EarlyStopping(monitor='val_loss',

                          mode='min',

                          patience=patience_earlystop,

                          verbose=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=patience_ReduceLROnPlateau, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=1e-5)
# Hyper Parameter

batch_size = 64

epochs = 50

history = model.fit(x=xtrain,

                    y=y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    validation_split=0.15,

                    verbose=1,

                    callbacks=[earlystop, learning_rate_reduction, mcp])
#Define a smooth function to display the training and validation curves

def plot_smoothed_learning_curves(history):

    val_loss = history.history['val_loss']#[-30:-1] #Uncomment if you want to see only the last epochs

    loss = history.history['loss']#[-30:-1]

    acc = history.history['acc']#[-30:-1]

    val_acc = history.history['val_acc']#[-30:-1]

    

    epochs = range(1, len(acc)+1 )

    

    # Plot the loss and accuracy curves for training and validation 

    fig, ax = plt.subplots(2,1, figsize=(12, 12))

    ax[0].plot(epochs, smooth_curve(loss), 'bo', label="Smoothed training loss")

    ax[0].plot(epochs, smooth_curve(val_loss), 'b', label="Smoothed validation loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)

    ax[0].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')



    ax[1].plot(epochs, smooth_curve(acc), 'bo', label="Smoothed training accuracy")

    ax[1].plot(epochs, smooth_curve(val_acc), 'b',label="Smoothed validation accuracy")

    legend = ax[1].legend(loc='best', shadow=True)

    ax[1].set_xlabel('Epochs')

    ax[1].set_ylabel('Accuracy')

    return



def smooth_curve(points, factor=0.8):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous*factor + point*(1-factor))

        else:

            smoothed_points.append(point)

    return smoothed_points
# Visualisation:

plot_smoothed_learning_curves(history)
#Load the best weights:

model.load_weights('best_weight.h5')
def plot_confusion_matrix(model_name):

    # Predict the values from the validation dataset

    y_pred = model_name.predict(xtest)

    # Because Y_pred is an array of probabilities, we have to convert it to one hot vectors 

    y_pred = np.argmax(y_pred,axis = 1)

    #Compute and print the accuracy scores:

    print('accuracy score:', accuracy_score(y_test,y_pred))

    # compute the confusion matrix 

    # By definition a confusion matrix C is such that C_i,j is equal to the number of observations known to be in group i but predicted to be in group j.

    cm = confusion_matrix(y_test, y_pred)

    cm = pd.DataFrame(cm, index = range(10), columns = range(10))

    # plot the confusion matrix

    plt.figure(figsize=(8,8))

    sns.heatmap(cm, cmap="Reds", annot=True, fmt='.0f')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

    return



plot_confusion_matrix(model)