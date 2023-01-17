# Data Wrangling

import numpy as np

import pandas as pd



# Visualizations

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from IPython.display import clear_output

plt.rcParams['figure.figsize'] = (12,6)

%matplotlib inline



# Neural Network Model (Keras with TensorFlow backend)

from keras.models import Sequential

from keras.optimizers import RMSprop#, Adadelta # removed due to bad performance

from keras.callbacks import ReduceLROnPlateau

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



# Model Evaluation

np.random.seed(2)

import itertools

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import scikitplot as skplt



clear_output()



print('Packages properly loaded!')
# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



print('train.shape, test.shape')

train.shape, test.shape
print('train dataset')

display(train.head(2))

print('test dataset')

display(test.head(2))
i = 0

n_rows, n_cols = 2, 6

fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12,4))

print(f'First {n_rows*n_cols} Digits of Training Dataset')

for row in range(n_rows):

    for col in range(n_cols):

        ax[row, col].imshow(train.iloc[i,1:].values.reshape(28,28))

        ax[row, col].set_title(f"Label {train.label.iloc[i]}")

        i+=1

plt.tight_layout(pad=2)
test.describe()
print('Pixel missing values')

print(f'test set: {test.isna().any().sum()}')

print(f'train set: {train.isna().any().sum()}')
# Let's check memory usage

train.info()
# If condition to ensure cell execution to be idempotent

if 'train' in globals(): 

    y_train = train["label"]

    X_train = train.drop(labels = ["label"], axis=1) 

    del train 



_ = y_train.value_counts().sort_index().plot(kind='bar', title='Sample Count per Digit', figsize=(12,4))
y_train = to_categorical(y_train, num_classes = 10)

print(f'Vector for first digit of training set (label 1): {y_train[0]}')

print(f'y_train.shape                                   : {y_train.shape}')
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0



# Reshape for CNN

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



X_train.shape, test.shape
display(X_train[0][:,:,0].shape)

_ = plt.imshow(X_train[0][:,:,0])
X_train.shape, y_train.shape
# Random seed for reproducibility

random_seed = 2



# Split into train and validation (since test definition is already attributed to submission data)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding='Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding='Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding='Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding='Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



epochs = 10

batch_size = 86
datagen = ImageDataGenerator(

        featurewise_center=False,  # turns off centering (mean=0) by feature

        samplewise_center=False,  # turns off centering (mean=0) by sample

        featurewise_std_normalization=False,  # turns off normalization by feature

        samplewise_std_normalization=False,  # turns off normalization by sample

        rotation_range=10,  # randomly rotates image from 0 to 10 degrees

        zoom_range = 0.1, # randomly zooms image by 10%

        width_shift_range=0.1,  # randomly shifts image horizontally by 10% of total width

        height_shift_range=0.1,  # randomly shifts image vertically by 10% of total height

        horizontal_flip=False,  # turns off random image flipping

        vertical_flip=False)  # turns off random image flipping



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
y_pred = model.predict(X_val)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(y_val,axis = 1) 



skplt.metrics.plot_confusion_matrix(y_true, y_pred_classes)
# Errors are difference between predicted labels and true labels

errors = (y_pred_classes - y_true != 0)



y_pred_classes_errors = y_pred_classes[errors]

y_pred_errors = y_pred[errors]

y_true_errors = y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)

most_important_errors = sorted_dela_errors[-6:]

display_errors(most_important_errors, X_val_errors, y_pred_classes_errors, y_true_errors)

plt.tight_layout(pad=2)
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)