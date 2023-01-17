#import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
from sklearn.metrics import confusion_matrix #visualization of results
import itertools
%matplotlib inline
# load dataset (Train)
train = pd.read_csv("../input/train.csv")
print(train.shape) #size of dataset

# load dataset (Test)
test= pd.read_csv("../input/test.csv")
print(test.shape)

# initialize variables for training
x_train = (train.iloc[:,1:].values).astype('float32') # only pixel values (no label)
y_train = (train.iloc[:,0].values).astype('int32') # only labels (digit corresponding to sample)
x_test = test.values.astype('float32') 

# normalize data
# pixels are in range [0-255], but the NN converges faster with smaller values [0-1]
x_train = x_train/255.0
x_test = x_test/255.0
X_train = x_train.reshape(x_train.shape[0], 28, 28,1)
X_test = x_test.reshape(x_test.shape[0], 28, 28,1)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

batch_size = 64 #number of samples per batch
num_classes = 10 # amount of recognizable digits (0-9) 
epochs = 20 # number of iterations
input_shape = (28, 28, 1)

# convert to binary class matrices (One Hot Encoding)
# example: 2 -> [0,0,1,0,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, num_classes)

# fix random seed for reproducibility
seed = 2
np.random.seed(seed)

#split training set and validation set for fitting (avoid overfitting)
#10% for the validation set and 90% is used to train the model
#The NN is then trained with the remaining of the training data, and in each epoch, the NN is tested against the validation data and we can see its performance.
#That way we can watch how the loss and accuracy metrics vary during training (e.g. bad if loss significantly smaller than val_loss, suggests overfitting)
#random_state ensures that the data is pseudo-randomly divided.
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = seed)
# define and add layers to model
model = Sequential()
# declare input layer --> convolution layer with 32 filters
# 'relu' is the rectifier (activation function max(0,x)). The rectifier activation function is used to add non linearity to the network.
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Dropout(0.25))
# weights from the Convolution layers must be flattened (1-dimensional) before passing them to the fully connected Dense layer.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

#loss function + optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

#data augmentation to prevent overfitting
#here: 
#randomly rotate some training images by 10 degrees, 
#randomly Zoom by 10% some training images
#randomly shift images horizontally by 10% of the width
#randomly shift images vertically by 10% of the height

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
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

# Look at confusion matrix 
#Note, this code is taken straight from the SKLEARN website, an nice way of viewing confusion matrix.
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
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))
#get the predictions for the test data
predicted_classes = model.predict_classes(X_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})
submissions.to_csv("submission_predictions.csv", index=False, header=True)

print(submissions.shape)
submissions.head()
