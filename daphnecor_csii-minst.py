# import all libraries 
import pandas as pd # for displaying and importing the data
import numpy as np # for the linear algebra (matrix operations)
import matplotlib.pyplot as plt # plotting
import matplotlib.image as mpimg # image plotting
import seaborn as sns # to make our plots pretty
%matplotlib inline

np.random.seed(2) # setting the starting parameters

from sklearn.model_selection import train_test_split # used to split training set in validation and training set
from sklearn.metrics import confusion_matrix # to make the confusion matrix
import itertools # for efficient iteration

# using keras 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # to build a NN
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D # methods used from the keras framework
from keras.optimizers import SGD # Stochastic Graidient Descent
from keras.callbacks import ReduceLROnPlateau # annealing LR

# Seaborn specifications
sns.set(color_codes=True) 
sns.set(style='white', context='notebook')
sns.set_palette("Blues",10)
# First we load the data using pandas
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

g = sns.countplot(Y_train)

Y_train.value_counts()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# Reshape image to 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# Now we encode the categorical labels to one hot vectors (e.g. : 3 -> [0,0,0,1,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Set random seed
random_seed = 2
# Split the training set to train and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# example images
image = plt.imshow(X_train[6][:,:,0])
model = Sequential() # to build the NN

# 2 Conv Layers, followed by the relu activation function and maxpooling
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size =(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size =(2, 2))) 
model.add(Dropout(0.25))

# flatten the data to 1 dimensional data
model.add(Flatten())

# Fully connected layer 1
model.add(Dense(1000, activation='relu'))

# Fully connected layer 2, this is the output layer
model.add(Dense(10, activation='softmax'))

model.summary() # give use a neat summary of the model
# First we define the optimizer we will use 
optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# Then compile the model (note: step 2 in the Keras workflow process)
# in this part we configure the learning process
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 10
batch_size = 16
# before training we want to specify the batch sizes the network will be trained on and the epochs
# model.fit(data, labels, epochs, batch_size)
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
                    validation_data = (X_val, Y_val), verbose = 2)
# Lets train!
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='r', label="Training loss") # a blue line for the Training loss
ax[0].plot(history.history['val_loss'], color='b', label="Validation loss",axes =ax[0]) # red for the validation loss
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='r', label="Training accuracy") 
ax[1].plot(history.history['val_acc'], color='b',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Plot the confusion matrix using the scikit learn library

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):   
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

""" Compute the Confusion Matrix """

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
# range is the number of classes we have
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Lets display some correct results

# if there is an error, or in other words if the difference between the prediction and the true label > 0
correct = (Y_pred_classes - Y_true == 0) # define what is correct

# specify the correct ones from each set
Y_pred_classes_correct = Y_pred_classes[correct]
Y_pred_correct = Y_pred[correct]
Y_true_correct = Y_true[correct]
X_val_correct = X_val[correct]

def display_correct(correct_index,image_correct,pred_correct, obs_correct):
    """ showing 6 images that were correctly classified"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            correct = correct_index[n]
            ax[row,col].imshow((image_correct[correct]).reshape((28,28)))
            ax[row,col].set_title("Predicted label - {}\n True label - {}".format(pred_correct[correct],obs_correct[correct]))
            n += 1
            
# Probabilities of the wrong predicted numbers
Y_pred_correct_prob = np.max(Y_pred_correct,axis = 1) # np.max flattens the nD array and returns the maximum

# Predicted probabilities of the true values in the error set
true_prob_correct = np.diagonal(np.take(Y_pred_correct, Y_true_correct, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_correct = Y_pred_correct_prob - true_prob_correct

# Sorted list of the delta prob errors
sorted_delta_correct = np.argsort(delta_pred_true_correct)

# Take the first 6 elements from the top errors
most_important_correct = sorted_delta_correct[-6:]

# Display the top 6 errors
display_correct(most_important_correct, X_val_correct, Y_pred_classes_correct, Y_true_correct)

plt.tight_layout() # make sure the padding around each images is sufficient
# Lets display some error results

# if there is an error, or in other words if the difference between the prediction and the true label > 0
errors = (Y_pred_classes - Y_true != 0)

# pick the errors from each set
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,image_errors,pred_errors, obs_errors):
    """ showing 6 images that were incorrectly classified"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((image_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label - {}\n True label - {}".format(pred_errors[error],obs_errors[error]))
            n += 1
            
# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1) # np.max flattens the nD array and returns the maximum

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_delta_errors = np.argsort(delta_pred_true_errors)

# Take the first 6 elements from the top errors
most_important_errors = sorted_delta_errors[-6:]

# Display the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

plt.tight_layout() # make sure the padding around each images is sufficient


