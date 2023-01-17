import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import SGD, Adam, RMSprop, Nadam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

import warnings

warnings.filterwarnings("ignore")



sns.set(style='white', context='notebook', palette='deep')





def binary_pred_stats(ytrue, ypred, threshold=0.5):

    one_correct = np.sum((ytrue==1)*(ypred > threshold))

    zero_correct = np.sum((ytrue==0)*(ypred <= threshold))

    sensitivity = one_correct / np.sum(ytrue==1)

    specificity = zero_correct / np.sum(ytrue==0)

    accuracy = (one_correct + zero_correct) / len(ytrue)

    return sensitivity, specificity, accuracy



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
# Load the data

df_train = pd.read_csv("../input/digit-recognizer/train.csv")

df_test = pd.read_csv("../input/digit-recognizer/test.csv")



# Input data should be 28x28 pixel images of numbers 0-9. 

# From the data description we know that the dataframes have the first column as the target, and the subsequent 784 pixel values
df_train.info() # we see 42000 entries of (1+784) columns, which are 42k training data points.

df_train.head()



Y_train = df_train['label'] # target data is the 0'th column

X_train = df_train.drop(labels = ["label"],axis = 1) # Input data are the remaining columns

X_test = df_test # input for the testing data is just the entire df_test dataframe (pixelvalues without labels)



# how are the distinct number distributed in our training data?

Y_train.hist() # pretty uniformly distributed

Y_train.value_counts() # quantitative measures



# We need to reshape the input data before we can use a CNN on it. We want it to be reshaped to 28x28 matrices for each entry

#reshape(dim1, dim2, dim3, channels)

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



input_shape = (28,28,1)





# Normalize from [0:255] to [0:1] for better computational performance

X_train = X_train / 255.0

X_test = X_test / 255.0





# Now we need to turn our labels (0,1,2,...) into categories by one-hot enconding.

Y_train = to_categorical(Y_train, num_classes = 10)





# Now we will split our training set into an actual training set and a validation set

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.15, random_state = 3) # all else equal, 0.15 gives better results than 0.10



# Check whether our preprocessing is ok

plt.figure()

g = plt.imshow(X_train[10][:,:,0])

print(Y_train[10])

# Define the CNN model

model = Sequential()



# first conv layer

model.add(Conv2D(filters = 128, kernel_size=(5, 5), activation='relu', padding='same', input_shape = input_shape))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size=(5, 5), activation='relu', padding='same', input_shape = input_shape))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

          

# Second conv layer

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



# Fully connected MLP layers

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



# Output layer

model.add(Dense(10, activation='softmax')) # for binaryclassification it should be 1, activation = 'sigmoid'



# We use cross entropy error and the adam or RMSprop optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) # RMSprop

# optimizer = Adam(lr=0.005)  # ADAM



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()
# 1 Set a learning rate annealer: How to handle plateaus in parameter space

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# 2 Epochs and batchsize

epochs = 50 # Cannot be too many to prevent overfitting

batch_size = 128





# 3 Data augmentation

# With data augmentation to prevent overfitting (accuracy 0.99286)



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.10, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)

%%time

# Now train the model

'''

estimator = model.fit(X_train, Y_train, 

                      validation_data=(X_val, Y_val),

                      epochs=50, 

                      batch_size=50,

                      verbose=0)

'''

# more advanced fit with data augmentation

estimator = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])



# Plot the training error

plt.plot(estimator.history['loss'])

plt.plot(estimator.history['val_loss'])

plt.title('Model training')

plt.ylabel('training error')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc=0)

plt.show()
# Get the training predictions and results for those

predtrain = model.predict(X_train)

sensitivity, specificity, accuracy = binary_pred_stats(Y_train, predtrain)

print("train set:", sensitivity, specificity, accuracy)



# Get the test predictions and the results for those

predtest = model.predict(X_val)

sensitivity, specificity, accuracy = binary_pred_stats(Y_val, predtest)

print("test set: ", sensitivity, specificity, accuracy)
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
# predict results

results = model.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_with_datagen.csv",index=False)