# Load Libraries and data provided by Kaggle.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

import random
# import BatchNormalization
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')



# Kaggle note:
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
#+ The current working directory
print("Kaggle working directory is", os.getcwd())
print("Output generated by this notebook will go here")
print("Current contents:",os.listdir(os.getcwd()))
#+ Set random seed for reproducibility
random.seed(10)
print("Setting randomization seed for results reproducibility...\n", "You should see a '74' here -> ", random.randint(1,101))
#+ the number "74" should display everytime this kernel is run
# Load the data.  Data files were automatically placed when the competiion was joined and this kernel was started.
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
# Get dimensions of the datasets
print("Shape of train =", train.shape)
print("Shape of test =", test.shape)
# Peek at a tiny part of both datasets
print("Training set:\n", train.iloc[31000, :10])
print("Testing set:\n", test.iloc[26000, :10])

# Build a -list- of just the labels in the training set
Y_train = train["label"]

# Drop 'label' column from the training set
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
# del train 

# Show a colorful plot comparing the number of examples of each digit 0-9 in the training set
g = sns.countplot(Y_train)

# Show the actual counts of each digit in the training set
Y_train.value_counts()

# Check the training data
print(X_train.isnull().any().describe())

# Are there any nulls
# a if condition else b
'There are nulls present in the training set' if (X_train.isnull().any().any()) else 'There are NO nulls present in the training set'

# Check the test data
print(test.isnull().any().describe())

# Are there any nulls
'There are nulls present in the test set' if print(test.isnull().any().any()) else 'There are NO nulls present in the test set'
# Perform a grayscale normalization to reduce the effect of illumination's differences.
# The CNN converg faster on [0..1] data than on [0..255].
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
# Basically all we are doing here is adding the channel dimension 1.  Greyscale uses 1 channel.  RGB uses 3
# Its needed by the NN algorithms
X_train = X_train.values.reshape(-1,28,28,1)  ## the -1 is a placeholder for the number of examples. 
#+ -1 is sort of a wildcard in this sense.
test = test.values.reshape(-1,28,28,1)
#+ Show an example image from the training set
g = plt.imshow(X_train[1000][:,:,0])
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Choose a seed value for reproducible results
random_seed = 2

# Do the split the train and the validation set 
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

# -train_test_split- (from sklearn.model_selection, scikit-learn package) is a quick utility that wraps input 
# validation # and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting 
# (and optionally subsampling) data in a oneliner.

# Using a 90/10 split here. 90% train/10% validation.

# Some examples
g = plt.imshow(X_train[0][:,:,0])
### A "Sequential" model in Keras model is a linear stack of layers.
model = Sequential()

### According to Andrew Ng, a common pattern you see in convolutional neural networks is:
### Conv -> Conv -> Pool -> Conv -> Pool -> FC -> FC -> FC -> Softmax

### The first convolutional layer
model.add(Conv2D(filters = 32, 
                 kernel_size = (5,5),
                 padding = 'Same', 
                 activation ='relu', 
                 input_shape = (28,28,1)))  # need to implicitly specify the dimensions of the example image because, don't forget, the image
# has been reshaped into a one-dimensional vector for input into the NN.

#+ Convolutional operations are the application of filters (small matrices) to the input values.  Their purpose is to detect -features- within the
#+ image.  They can also have the effect of reducing the size of the image and thereby shrinking the number of parameters that need to be updated during gradient
#+ descent calculations. This can be important with large images having multiple channels (eg. red-blue-green). The images here are relatively small so parameter 
#+ reduction is not particularly important.  That's why in this case the "padding" parameter is set to "Same" causing the output to be the same size as the 
#+ input; no size reduction.  

#+ Each cell within the filter has a value, or "weight", that is used to smooth and summarize the input in order to find features.  The filters in the first 
#+ layer are 5 by 5 in size.  In a Keras implementation the weights are automatically randomly initialized and over time will gradually adjust into values 
#+ that can identify features, like edges, within the image.

#+ Every layer in a NN must be associated with a non-linear "activation" function. This function is applied to the product of the current layer's weight and the previous 
#+ layer's activation value to generate the next activation value which will in turn be passed to the next layer in the NN.  The first convolutional layer uses
#+ -relu- as its non-linear function. Note - the first set of activation values are by default simply the pixel values of the original image.

#+ Using non-linear activation functions are what make NNs work.  It can be demonstrated (not here) that if a -linear- activation function is used, it won't 
#+ matter how complex the NN is in terms of the number of layers, the prediction result will be the same as if there were only one layer.  

### Add another convolutional layer.
model.add(Conv2D(filters = 32, 
                 kernel_size = (5,5),
                 padding = 'Same', 
                 activation ='relu'))
#+ This further refines the low level feature detection.
### Add a pooling level.
model.add(MaxPool2D(pool_size=(2,2)))

#+ Pooling is a way of reducing the size of the image and speeding up the processing of computation.  It makes the recognition of features more robust.
#+ This layer uses -max pooling- size 2 by 2, which means for every 2 by 2 region in the previous layer the maximum value is selected and passed on as the new
#+ value.  There is no good theoretical underpinning for pooling other than it provides good results.  What it is doing is finding the most distinctive features.
#+ Since no "stride" parameter is specified, the pooling window will be moved in increments of 2 pixels across and then down, then across, etc until the 
#+ entire image is scanned.  The result will be a 7 by 7 image (28 by 28 image divided into 2 by 2 regions condenses to 7 by 7)
### Add a "drop out" layer.
model.add(Dropout(0.25))

#+ "Drop out" ignores a percentage of the activations coming from the previous layer. In this case 1 out of 4 will be ignored.  This is a form of regularization
#+ used to prevent overfitting.  It helps in ignoring noise in the image and forces the remaining activations "to work harder".
### Add another convolutional layer
model.add(Conv2D(filters = 64, 
                 kernel_size = (3,3),
                 padding = 'Same', 
                 activation ='relu'))
#+ The filter size is 3 by 3 and there are 64 filters.  
### Another convolutional layer
#+ For Experiment 7 do not use this second 64 filter layer
#+ model.add(Conv2D(filters = 64, 
#+                  kernel_size = (3,3),
#+                  padding = 'Same', 
#+                  activation ='relu'))
### Add a pooling layer
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
### Another drop out layer
model.add(Dropout(0.25))
### Reshape the 2d data to a one dimensional vector
model.add(Flatten())
### Add a regular neural network layer with 256 nodes
model.add(Dense(256, activation = "relu"))
#+ This layer will be "fully connected" to the previous layer, meaning every activation value in the previous
#+ layer will be passed to every node in the "Dense" layer.

#+ For Experiment 2-7, add a batch normalization layer.
model.add(BatchNormalization())
#+ Batch normalization standardizes the mean and variance of Z values prior to an activation layer.
#+ It works with gradient descent, RMSProp, Adam, momentum and mini batch.
#+ The normalization parameters Beta and Gamma are learned during training.  These determine the mean and
#+ variance that the Z values will be constrained by.  Used as a defense against "covariant shifting".
### Add another regular neural network layer with 128 nodes
#+ For Experiment 6 use this additional fully connected layer. For Basic model and all other experiments comment this out.
#+ model.add(Dense(128, activation = "relu"))
#+ This layer will be "fully connected" to the previous layer, meaning every activation value in the previous
#+ layer will be passed to every node in the "Dense" layer.
### Add a drop out layer that will randomly select half the nodes in the Dense layer to be ignored.
model.add(Dropout(0.5))  #+ Basic model and Experiment 5, 6 uses this drop out layer
#+ Exp 1, Exp 2, Exp 3, Exp 4: eliminate the final drop out layer by commenting out the previous line.
#+ My thinking about taking out the drop out layer is that it is usually used as a regularization technique to 
#+ prevent over fitting.  There didn't seem to be over fitting going on so I took it out. Accuracy seemed to
#+ improve without it.

### Add a final fully connected Dense layer where the final classification step will take place
model.add(Dense(10, activation = "softmax"))
#+ Activation is "Softmax" which offers a probability of each digit value being the input image.
### Define the optimizer using RMSProp
#+ For Basic Model and Experiment 1 - 3 and 6 use RMSProp.
my_optimizer = RMSprop(lr=0.001, 
                    rho=0.9, 
                    epsilon=1e-08, 
                    decay=0.0)
#+ The optimizer function has influence over the effectiveness of the process of arriving at the final solution. 
#+ The "learning rate" (LR) is a multiplier factor that determines the size of the change in the weight parameters
#+ during each backpropagation step.  A larger LR speeds up learning but too large can lead to wild occilations in the 
#+ gradient descent process and failure to find an optimal solution. Too small a learning rate 
#+ can cause the learning process to proceed too slowly and maybe never reaching an optimal solution.
#+ 'rho' is a constant typically set to 0.9 or 0.99 is a dampening factor for how gradient descent occilates during learning.
#+ It has the effect of speeding the path taken to finding the optimal solution.
#+ 'epsilon' is an arbitrary tiny, near zero, value that is added to prevent division by zero in the calculations of the gradient descent delta.
#+ 'decay' is a factor that decreases the learning rate periodically.  It is set to zero here.  A learning rate decay factor is applied
#+ in a later step in a more effective way.

### Define the optimizer using Adam (Adaptive Movement Estimation)
#+ For Exp 4 and 5 use Adam.
#+ my_optimizer = Adam(lr=0.001, 
#+                     beta_1 = 0.09, 
#+                     beta_2 = 0.999, 
#+                     epsilon = 1e-08, 
#+                     decay = 0.0, 
#+                     amsgrad = False)
#+ lr: float >= 0. Learning rate.
#+ beta_1: float, 0 < beta < 1. Generally close to 1.
#+ beta_2: float, 0 < beta < 1. Generally close to 1.
#+ epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
#+ decay: float >= 0. Learning rate decay over each update.
#+ amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".


### Compile the model with the optimizer defined above.  
model.compile(optimizer = my_optimizer, 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"])
# Using a cost function called "categorical cross entropy" that is appropriate for multi-class classification.

### Set the number of epochs for training.  
#+ 1 epoch used here just for a reality check that the model can be trained successfully.
#+ Will switch to using more epochs for the real model.
my_epochs = 1 
my_batch_size = 86

import datetime
# Train the model with the training data.
# In Keras, training returns a "history" object which contains information about the accuracy of the fitting process.
print(datetime.datetime.now())
history = model.fit(X_train, 
                    Y_train, 
                    batch_size = my_batch_size, 
                    epochs = my_epochs, 
                    validation_data = (X_val, Y_val), 
                    verbose = 2)
print(datetime.datetime.now())
#+ One epoch takes three minutes without GPU. 10 seconds with GPU.
#+ Good demo of the value of faster computer hardware.
#+ 30 epochs took 3 minutes with the GPU.
# Try adding more data using data augmentation.  
#+ Training with more data is one way to reduce overfitting.
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
#+ The ImageDataGenerator is a terrific feature in Keras as it simplifies into one step the generation of
#+ additional image data to increase the number of training examples. It comes with a large number of parameters 
#+ for distorting images, so #+ examine them carefully in the Keras documentation to find the ones you 
#+ will explicitly set and which you will leave at their default values.

# Fits the data generator to the example data
datagen.fit(X_train)

# Set a learning rate annealer.
#+ Use the change in the validation accuracy to control the speed of LR reduction.
#+ don't let the learning rate drop below 0.00001.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
#+ The term "annealing" has multiple definitions.  It's commonly used to describe a process of hardening metal 
#+ or glass by controlling the speed at which it cools from its molten state.  There are also meanings within 
#+ biology and mathematics.  The deep learning meaning has to do with slowing the speed with which the learning 
#+ rate changes.  Extra Note - "Simulated annealing" is related to stochastic gradient descent.  It is used 
#+ to find a satisfactory approximate solution in solution spaces that are too large to find an exact solution.

#+ Increase the number of epochs to 30.  Use GPU if available.
#+ Experiment 3 - 7. 100 epochs
#+ my_epochs = 100
my_epochs = 1  ### The Kaggle Kernel concept is great but could use some embelishments such as the ability to cache results or the ability to disable cells temporarily.
#+ The epoch count is set to "1" here in order to allow the program to more quickly commit and produce the final ensemble result using previously generated results.

# Fit the model using the augmented data generator.  
#+ Make predictions for the validation set.
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = my_batch_size),
                              epochs = my_epochs, 
                              validation_data = (X_val,Y_val),
                              verbose = 2, 
                              steps_per_epoch = X_train.shape[0] // my_batch_size, 
                              callbacks=[learning_rate_reduction])
#+ the .flow method takes data & label arrays and generates batches of augmented data while learning takes 
#+ place

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
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
### End of function definition
    
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
#+ Y_pred contains a list of probabilities for each image. There are 10 probabilities for each image corresponding to
#+ the probability that the image represents a 0, 1, 2... 9.  The largest probability in the list is the number 
#+ that will be declared the predicted value.
#+ Example for the first image: Y_pred[1] = [1.2758733e-13 2.1202715e-10 2.4014818e-07 4.6213042e-10 5.0552443e-02
#+ 9.5689595e-11 5.5256303e-14 8.5465127e-04 3.7065365e-06 9.4858891e-01] 
#+ 9.4858891e-01 is the largest value, so the predicted value for the image is "9".

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
#+ Y_pred_classes are the digits that were predicted for each image.  Eg: [6 9 5 ... 2 2 6]
#+ Y_true are the actual digit (class) for each image.  Eg: [6 9 5 ... 2 2 6]
errors = (Y_pred_classes - Y_true != 0)
#+ errors is a list of True & False values, one for each image in the validation set.
#+ False means there was no error and the predicton was correct. True means the prediction was wrong. 
#+ When the difference between Y_pred_classes - Y_true is equal to zero, the expression is False and will be 
#+ ones we got right. 
#+ Instances where Y_pred_classes - Y_true is not zero, the expression is True and will be the ones we got wrong.

Y_pred_classes_errors = Y_pred_classes[errors]
#+ This pulls out just the predictions that we got wrong (the "True" values in errors).  
#+ It is a list of all the mistakes.
#+ Example: [8 9 9 6 8 6 8 8 7 9 3 6 7 8 4 4 9 6 0]

Y_pred_errors = Y_pred[errors]
#+ Y_pred[errors] gets a list of all the probabilities that were made for the errors.  There is a probability 
#+ for each class of possible outcomes.

Y_true_errors = Y_true[errors]
#+ The true labels of the ones we got wrong.

X_val_errors = X_val[errors]
#+ X_val_errors contains pixel values for the errors. So they can be displayed graphically to see what the 
#+ hard to predict digits look like.

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images that had prediction errors with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex = True, sharey = True)
    fig.subplots_adjust(hspace = .5)  #+ adds space between rows of subplots so captions are all visible.
    #+ Citation: https://stackoverflow.com/questions/5159065/need-to-add-space-between-subplots-for-x-axis-label-maybe-remove-labelling-of-a 
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label: {}\nTrue label: {}".format(pred_errors[error],obs_errors[error]))
            n += 1  #+ increment counter n

# Probabilities of the wrong predicted numbers.
#+ In the final Softmax layer, every class of outcome (0, 1,...9) has a probability calculated, for each input image.
#+ Y_pred_errors contains a 2 dimensional matrix of all the prediction probabilities for the validation images the model got wrong. 
#+ Each row has the calculated probability for each possible class outcome (0...9)
#+ Y_pred_errors_prob contains the probabilties that the model calculated for each incorrectly predicted class.  It contains the maximum
#+ probability among the 10 classes 
Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1)

# Predicted probabilities of the true values in the error set.
#+ true_prob_errors contains the probabilty the model calculated for the correct class.
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

#+ Calculate the difference between the probability that was assigned to the incorrectly predicted label and the 
#+ probability that was assigned to the true label.  A larger difference means the model got it "more wrong".
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
#+ Since we are only dealing with the errors instances here, by definition Y_pred_errors_prob is greater 
#+ than true_prob_errors for a given image.

# Sorted list of the delta prob errors
#+ Argsort sorts in ascending order
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
#+ Use -6 to select the bottommost 6 items in the sorted list.  These will be the 6 with the greatest error.
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
#+ Using the display_errors() function defined above.
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
# predict results
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
#+ These are all commented out when subm,itting ensemble. See below.
#+ submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

#+ Put a copy of the submission in the Kaggle workoing directory
#+ submission.to_csv(path_or_buf = "cnn_mnist_datagen.csv",index=False)

#+ Load the three prediction files needed for the ensemble.  It assumes these have been previously uploaded to the ../input folder in Kaggle.
Exp03 = pd.read_csv("../input/ensemble-input/cnn_mnist_datagen_Experiment_03.csv")
Exp06 = pd.read_csv("../input/ensemble-input/cnn_mnist_datagen_Experiment_06.csv")
Exp07 = pd.read_csv("../input/ensemble-input/cnn_mnist_datagen_Experiment_07.csv")

#+ Confirm the shapes. They should all be (28000, 2)
assert Exp03.shape == (28000, 2)
assert Exp06.shape == (28000, 2)
assert Exp07.shape == (28000, 2)

#+ Change column names
Exp03.columns = ['ImageId', 'Exp03']
Exp06.columns = ['ImageId', 'Exp06']
Exp07.columns = ['ImageId', 'Exp07']

#+ Concatenate the three dataframes lable columns.
EnsembleDF = pd.concat([Exp03, Exp06[['Exp06']], Exp07[['Exp07']]], axis=1)

#+ Ensemble shape should be (28000, 4)
assert EnsembleDF.shape == (28000, 4)

#+ Ensemble column names should be ['ImageId', 'Exp03', 'Exp06', 'Exp07']
assert list(EnsembleDF) == ['ImageId', 'Exp03', 'Exp06', 'Exp07']

#+ Exp03 will be the baseline value. If the modal value (most frequent value) for a row is different from Exp03, use the modal value instead.
modevalues = pd.DataFrame(EnsembleDF.mode(axis=1, numeric_only=False))

ModeValuesDF = modevalues
ModeValuesDF.columns = ['ModalValue', '1', '2', '3']
list(ModeValuesDF)
#+ Concatenate the three dataframes lable columns.

EnsembleSubmission = pd.concat([Exp03['ImageId'], ModeValuesDF['ModalValue'].astype(int)], axis = 1)
EnsembleSubmission.columns = ['ImageId', 'Label']
EnsembleSubmission.to_csv(path_or_buf = "cnn_mnist_datagen_ensemble.csv",index=False)
print(EnsembleSubmission.shape)
print(EnsembleSubmission.head(100))

