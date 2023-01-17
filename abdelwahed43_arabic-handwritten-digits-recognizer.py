# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Import the necessary libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.datasets import mnist



sns.set(style='white', context='notebook', palette='deep')

# Load the data

X_train = pd.read_csv("../input/ahdd1/csvTrainImages 60k x 784/csvTrainImages 60k x 784.csv")

Y_train = pd.read_csv("../input/ahdd1/csvTrainLabel 60k x 1.csv")

test = pd.read_csv("../input/ahdd1/csvTestImages 10k x 784.csv")

y_test = pd.read_csv("../input/ahdd1/csvTestLabel 10k x 1.csv")

print(X_train.shape)

print(Y_train.shape)

print(test.shape)

print(y_test.shape)
#plot Count numbers

Y_train=Y_train.iloc[:,0]

y_test=y_test.iloc[:,0]

g = sns.countplot(Y_train)



Y_train.value_counts()



# Check the data

X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalize the data to make CNN faster

X_train = X_train / 255.0

test = test / 255.0

print(X_train.shape)

print(Y_train.shape)

# Reshape image is 3D array (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
print(X_train.shape)

print(Y_train.shape)



print(test.shape)

print(y_test.shape)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)

y_test  = to_categorical(y_test , num_classes = 10)
print(Y_train.shape)

print(y_test.shape)
# Set the random seed

random_seed = 2
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# Draw an example of a data set to see 



img_num=663   #  Please try with this number to understand how augment you data 

               # 1111, 111 ,101 ,144 ,663

g = plt.imshow(X_train[img_num][:,:,0])



print(Y_train[img_num])

#g = plt.imshow(X_train[0][:,:,0])
#Creating CNN model

"""

  [[Conv2D->relu]*2 -> BatchNormalization -> MaxPool2D -> Dropout]*2 -> 

  [Conv2D->relu]*2 -> BatchNormalization -> Dropout -> 

  Flatten -> Dense -> BatchNormalization -> Dropout -> Out

"""

model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Dense(10, activation = "softmax"))

model.summary()
#model.summary()
# print out model look

from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image("model.png")
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

# Audjusting learning rate

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 100 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 32 #64,128 ,256,512 
# Without data augmentation i obtained an accuracy of 0.98114

#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 

#          validation_data = (X_val, Y_val), verbose = 2)
# With data augmentation to prevent overfitting (accuracy 0.99286)

'''

Let's apply some data augmentation.



Data augmentation is a set of techniques used to generate new training samples from the original ones

by applying jitters and perturbations such that the classes labels are not changed.

In the context of computer vision, these random transformations can be translating,

rotating, scaling, shearing, flipping etc.



Data augmentation is a form of regularization because the training algorithm is being

constantly presented with new training samples,

allowing it to learn more robust and discriminative patterns

and reducing overfitting.

'''







datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# Fit the model

"""

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, 

                              shuffle = True,

                              validation_data = (X_val,Y_val),

                              verbose = 2, 

                              steps_per_epoch=X_train.shape[0] // batch_size+1,

                              validation_steps = X_val.shape[0] //batch_size+1,

                              callbacks=[learning_rate_reduction])

"""



#Prediction model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# Draw the loss and accuracy curves of the training set and the validation set.

# Can judge whether it is under-fitting or over-fitting

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 



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


# Show some wrong results, and the difference between the predicted label and the real labe

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
# predict results y_pred

y_pred = model.predict(test) 



print(test.shape)

print(y_test.shape)

print(y_pred.shape)
# Evaluate model

score = model.evaluate(test , y_test,verbose=3)



print('Test accuarcy: %2f%%' % round((score[1] * 100),5))

#print('Loss accuracy: %2f%%' % round((score[0] * 100),0))

#print(model.metrics_names)
# select the indix with the maximum probability

results = np.argmax(y_pred,axis = 1)



results = pd.Series(results,name="Label")
# Save the final result in cnn_mnist_submission.csv

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_submission.csv",index=False)