# Load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



# Set the random seed

np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

epochs = 5 # Number of iterations needed for the network to minimize the loss function

batch_size = 128 

num_classes = 10 # Total number of class labels (0-9 digits)
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Data exploration

train.head()
# Data exploration

test.head()
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 



# Frequency histogram of numbers in training data

label_count = sns.countplot(Y_train)
# Value counts

unique, counts = np.unique(Y_train, return_counts=True)

data=dict(zip(unique, counts))

data
# Free some space

del train
X_train.describe()
Y_train.describe()
# Normalizing the data

X_train = np.array(X_train / 255.0)

test = np.array(test / 255.0)

# Reshape image in 3 dimensions

X_train_view = X_train.reshape(X_train.shape[0], 28, 28)



# Displaying some data from the train set

plt.figure(figsize=(10, 10))

plt.subplots_adjust(top=0.5)



for i in range(10):

    plt.subplot(2, 5, (i+1))

    plt.imshow(X_train_view[i])

    plt.title('label:{}'.format(Y_train[i]));
X_train = X_train.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)

Y_train_data = Y_train



# One-hot vector as a label (binarize the label)

Y_train = to_categorical(Y_train, num_classes = 10)



# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=1)
Y_val.shape
# Displaying some data from the test set

X_test = np.reshape(np.array(test), (-1, 28, 28, 1))

plt.figure(figsize=(10,10))

plt.subplots_adjust(top=0.5)

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_val[i].reshape(28, 28))
# Building a linear stack of layers with the sequential model

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# Improves parameters to minimise the loss

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# With data augmentation to prevent overfitting

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
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),verbose = 2, 

                              steps_per_epoch=X_train.shape[0] // batch_size)
history.history.keys()
# evaluate the model

scores = model.evaluate(X_val, Y_val, verbose=2)
# history plot for accuracy

plt.plot(history.history["acc"])

plt.plot(history.history["val_acc"])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Training accuracy", "Test accuracy"], loc="upper left")

plt.show()
# history plot for accuracy

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Training loss", "Test loss"], loc="upper left")

plt.show()
print("Test loss:", scores[0])

print("Test accuracy:", scores[1])

# Confusion matrix

def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



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
correct_indices = np.nonzero(Y_pred_classes == Y_true)[0]

incorrect_indices = np.nonzero(Y_pred_classes != Y_true)[0]

print()

print("Classified correctly: ", len(correct_indices))

print("Classified incorrectly: ", len(incorrect_indices))
# adapt figure size to accomodate 12 subplots

plt.rcParams['figure.figsize'] = (7,14)



figure_evaluation = plt.figure()



# plot 6 correct predictions

for i, correct in enumerate(correct_indices[:6]):

    plt.subplot(6,3,i+1)

    plt.imshow(X_val[correct].reshape(28,28))

    plt.title(

      "Predicted: {}, Actual: {}".format(Y_pred_classes[correct],

                                        Y_true[correct]))

    plt.xticks([])

    plt.yticks([])



# plot 6 incorrect predictions

for i, incorrect in enumerate(incorrect_indices[:6]):

    plt.subplot(6,3,i+10)

    plt.imshow(X_val[incorrect].reshape(28,28))

    plt.title(

      "Predicted {}, Actual: {}".format(Y_pred_classes[incorrect], 

                                       Y_true[incorrect]))

    plt.xticks([])

    plt.yticks([])
