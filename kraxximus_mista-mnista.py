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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
y_train = train_df["label"]
x_train = train_df.drop(["label"], axis=1)
del train_df
# Count and visualize data
g = sns.countplot(y_train)
y_train.value_counts()
"""NORMALIZE VALUES TO (0 - 1.0)"""

x_train /= 255.0
test_df /= 255.0

"""RESHAPE TO (HEIGHT 28PX, WIDTH 28PX, CANAL 1)"""
# Keras requires an extra dimension at the end for channels. As our images are greyscale, we only require 1 channel (Boldness of black)
x_train = x_train.values.reshape(-1,28,28,1)
test_df = test_df.values.reshape(-1,28,28,1)

"""LABEL ENCODING"""
# Labels to one-hot vectors
y_train = to_categorical(y_train, num_classes=10)

"""SPLIT TRAINING & VALIDATION"""
random_seed = 2
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
"""BUILD THE CNN MODEL"""

model = Sequential() # Sequential model adds one layer at a time

model.add(Conv2D(filters=32,
                kernel_size=(5,5), # Size of 'filter'/'kernel'
                padding="same",
                activation="relu",
                input_shape=(28,28,1),
                ))
model.add(Conv2D(filters=32,
                kernel_size=(5,5),
                padding="same",
                activation="relu",
                input_shape=(28,28,1),
                ))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                kernel_size=(3,3),
                padding="same",
                activation="relu",
                ))
model.add(Conv2D(filters=64,
                kernel_size=(3,3),
                padding="same",
                activation="relu",
                ))
model.add(MaxPool2D(pool_size=(2,2),
                   strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten()) # Flatten to 1D
model.add(Dense(256,activation="relu")) # Typical NN layer
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))
"""BUILD LEARNING MODEL"""

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Categorical crossentropy is good for dem 1-hots
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001
                                           )
epochs = 1
batch_size = 86
"""DATA AUGMENTATION"""
# 2-3x dataset with augmentations to prevent overfitting
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


datagen.fit(x_train)
"""RUN IT"""

history = model.fit_generator(datagen.flow(x_train,
                                           y_train,
                                           batch_size=batch_size
                                          ),
                              epochs = epochs,
                              validation_data = (x_val,y_val),
                              verbose = 2,
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction]
                             )
"""PLOT DATA"""

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history["loss"], color="b", label="Training loss")
ax[0].plot(history.history["val_loss"], color="r", label="Validation loss", axes=ax[0])
legend = ax[0].legend(loc="best", shadow=True)

ax[1].plot(history.history["acc"], color="b", label="Training accuracy")
ax[1].plot(history.history["val_acc"], color="r", label="Validation accuracy")
legend = ax[1].legend(loc="best", shadow=True)
"""CONFUSION MATRIX"""

def plot_confusion_matrix(cm,
                         classes,
                         normalize=False,
                         title="Confusion matrix",
                         cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
# Predict the values from the validation dataset
y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
"""ERRORS"""
# Errors are difference between predicted labels and true labels
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
x_val_errors = x_val[errors]

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
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_delta_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_delta_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, x_val_errors, y_pred_classes_errors, y_true_errors)
"""GENERATE TEST RESULTS"""

results = model.predict(test_df)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
"""GENERATE CSV"""

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)


