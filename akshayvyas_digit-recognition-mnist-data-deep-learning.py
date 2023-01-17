import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')



np.random.seed(2)



import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
#Obtaining train data

train = pd.read_csv("../input/train.csv")

train.head(5)
#Obtaining test data

test = pd.read_csv("../input/test.csv")

test.head(5)
Y_train = train["label"]



#Now dropping the label column from the dataset

X_train = train.drop(labels = ["label"], axis = 1)



#Free memory space

del train



X_train.head(5)
#Creating a count plot for all the labels

sns.countplot(Y_train)

Y_train.value_counts()
#check missing values in train dataset

X_train.isnull().any().describe()
test.isnull().any().describe()
#Normalize the data

X_train = X_train / 255.0

test = test / 255.0
#Reshape the image in 3 dimensions (height = 28px, width = 28px, canal = 1))

# from 784 values in 1 Dimensional vector

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
# Encoding labels to one hot vectors

# Example: 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

from keras.utils.np_utils import to_categorical # convert-to-one-hot-encoding



Y_train = to_categorical(Y_train, num_classes = 10)

Y_train
#Spliting data into training and validation set

from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)
plt.imshow(X_train[0][:,:,0])
#Convolutional Neural Network Architecture

#CNN Model

#Input -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> 

#         Flatten -> Dense -> Dropout -> Output



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



def get_cnn_model():

    model = Sequential([

        Conv2D(filters = 32, kernel_size = (5,5), padding='Same',

              activation = 'relu', input_shape = (28, 28, 1)),

        Conv2D(filters = 32, kernel_size = (5,5), padding='Same',

              activation = 'relu'),

        MaxPool2D(pool_size = (2,2)),

        Dropout(0.25),

        Conv2D(filters = 32, kernel_size = (5,5), padding='Same',

              activation = 'relu'),

        Conv2D(filters = 32, kernel_size = (5,5), padding='Same',

              activation = 'relu'),

        MaxPool2D(pool_size = (2,2)),

        Dropout(0.25),

        Flatten(),

        Dense(256, activation = 'relu'),

        Dropout(0.5),

        Dense(10, activation = 'softmax')

    ])

    return model



model = get_cnn_model()
#Defining the Optimizer

from keras.optimizers import RMSprop



optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#Compiling the model

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", 

             metrics = ["accuracy"])
#Setting a learning rate annealer

from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',

                                           patience = 3, 

                                           verbose = 1,

                                           factor = 0.5, 

                                           min_lr = 0.00001)
epochs = 3

batch_size = 86
#Calculation Validation accuracy

history = model.fit(X_train, Y_train, batch_size = batch_size,

                   epochs = epochs,

                   validation_data = (X_val, Y_val), 

                   verbose = 2)
#Now performing Data Augmentation to improve Accuracy

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

            featurewise_center = False,

            samplewise_center= False,

            featurewise_std_normalization= False,

            samplewise_std_normalization= False,

            zca_whitening= False,

            rotation_range= 10,

            zoom_range= 0.1,

            width_shift_range=0.1,

            height_shift_range=0.1,

            horizontal_flip=False,

            vertical_flip=False)



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, 

                                          batch_size=batch_size),

                             epochs = epochs,

                             validation_data = (X_val, Y_val),

                             verbose =2, 

                             steps_per_epoch=X_train.shape[0] // batch_size,

                             callbacks = [learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 

from sklearn.metrics import confusion_matrix

import itertools



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



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)