# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Helper libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation,BatchNormalization,Conv2D,Dense,Dropout,Flatten,MaxPooling2D

from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint,EarlyStopping

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
print(f"Train shape: {train.shape}")
test.head()
print(f"Test shape: {test.shape}")
train_label = train["label"]

train_data = train.drop("label",axis = 1) 



sns.countplot(train_label)

train_label.value_counts()
# Check the data

train_data.isnull().any().describe()
# Check the data

test.isnull().any().describe()
# Normalize the data

X_train = train_data / 255.0

X_test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



print(f"X_train data shape: {X_train.shape}")

print(f"X_test data shape: {X_test.shape}")
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = tf.keras.utils.to_categorical(train_label, num_classes = 10)

print(f"Y_train data shape: {Y_train.shape}")
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=42)
def plot_random_digit():

    random_index = np.random.randint(0,X_train.shape[0])

    plt.imshow(X_train[random_index][:,:,0], cmap='gray')

    index = tf.argmax(Y_train[random_index], axis=0)

    plt.title(index.numpy())

    plt.axis("Off")
plt.figure(figsize=[2,2])

plot_random_digit()
# preview the images

plt.figure(figsize=[10,6])

for i in range(50):

    plt.subplot(5, 10, i+1)

    plt.imshow(X_train[i][:,:,0], cmap='gray')

    plt.axis('Off')
train_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180) 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        shear_range = 0.1, # Shear Intensity (Shear angle in counter-clockwise direction in degrees)

        zoom_range = 0.1, # randomly zoom image

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



train_datagen.fit(X_train)
model = Sequential([

    Conv2D(32, 5, padding='same', input_shape=(28, 28, 1)),

    Activation('relu'),

    

    Conv2D(32, 5, padding='same'),

    Activation('relu'),

    MaxPooling2D(2, 2),

    Dropout(0.25),

    

    Conv2D(64, 3, padding='same'),

    Activation('relu'),

    

    Conv2D(64, 3, padding='same'),

    Activation('relu'),

    MaxPooling2D(2, 2),

    Dropout(0.25),

    

    Flatten(),

    Dense(256),

    Activation('relu'),

    Dropout(0.5),

    Dense(10, activation="softmax"),

])

model.summary()
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
learning_rate=0.001

batch_size = 64

epochs = 50

steps_per_epoch = X_train.shape[0] // batch_size

validation_steps = X_val.shape[0] // batch_size
# Define the optimizer

optimizer = Adam(lr=learning_rate)
# Compile the model

model.compile(optimizer=optimizer,

              loss=categorical_crossentropy,

              metrics=['accuracy'])
def scheduler(epoch):

    return learning_rate * 0.99 ** epoch
lr_scheduler = LearningRateScheduler(scheduler)



model_checkpoint  = ModelCheckpoint('model_best_checkpoint.h5', save_best_only=True,

                                    save_weights_only=True, monitor='val_loss', verbose=1)



early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)



callbacks_list = [lr_scheduler, model_checkpoint, early_stopping]
# training the model

history = model.fit_generator(

      train_datagen.flow(X_train,Y_train,batch_size=batch_size),

      steps_per_epoch=steps_per_epoch,

      epochs=epochs,

      validation_data=(X_val,Y_val),

      validation_steps=validation_steps,

      verbose=1,

      callbacks=callbacks_list)
accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(len(accuracy))



plt.figure(figsize=(16, 4))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, accuracy, label='Training Accuracy')

plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
model.load_weights('model_best_checkpoint.h5')
results = model.evaluate(X_val, Y_val, batch_size=batch_size, verbose=0)



print(f"\nLoss: {results[0]}")

print(f"Accuracy: {results[1]}")
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



# Probabilities of the wrong predicted

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

results = model.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
# creating the submission file

submission = pd.DataFrame({"ImageId":[i+1 for i in range(len(X_test))],

                           "Label": results})

submission.head()
submission.to_csv("submission_5.csv", index=False, header=True)
# saving the model

model.save("MNIST_MODEL.h5")