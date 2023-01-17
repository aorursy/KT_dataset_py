import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
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
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 
g = sns.countplot(Y_train)
Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# # super simple data pre process
# scale = np.max(X_train)
# X_train /= scale
# test /= scale

# mean = np.mean(X_train)
# X_train -= mean
# mean = np.mean(test)
# test -= mean

# #visualize scales

# print("Max: {}".format(scale))
# print("Mean: {}".format(mean))
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
from keras.models import Model
from keras.layers import *

img_input=Input(shape=(28,28,1))
x=Conv2D(16, (3, 3), activation='relu')(img_input)
x=BatchNormalization()(x)
x =Activation('relu')(x)
x=Conv2D(32, (3, 3), activation='relu')(x)
x=BatchNormalization()(x)
x =Activation('relu')(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(64, (3, 3), activation='relu')(x)
x=BatchNormalization()(x)
x =Activation('relu')(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(128, (3, 3), activation='relu')(x)
x=BatchNormalization()(x)
x =Activation('relu')(x)
x=Flatten()(x)
x=Dense(64,activation='relu')(x)
x=Dense(10,activation='softmax')(x)

model=Model(img_input,x)
model.summary()
from keras.optimizers import SGD,Adam
from keras.callbacks import LearningRateScheduler, TensorBoard,ModelCheckpoint
#################optimization_lossfunction##############################
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

##################model saving########################################
checkpoint = ModelCheckpoint('weights.h5',  # model filename
                             monitor='val_acc', # quantity to monitor
                             verbose=1, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='max') # The decision to overwrite model is m
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

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
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=128),
                              epochs = 60, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // 128)
history_dict = history.history
history_dict.keys()
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
y_hat = model.predict(X_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(Y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)
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
# #get the predictions for the test data
# predicted_classes = model.predict(X_val)

# #get the indices to be plotted
# y_true = Y_val[:, 0]
# correct = np.nonzero(predicted_classes==y_true)[0]
# incorrect = np.nonzero(predicted_classes!=y_true)[0]
# from sklearn.metrics import classification_report
# target_names = ["Class {}".format(i) for i in range(10)]
# print(classification_report(y_true, predicted_classes, target_names=target_names))

# predicted = model.predict(X_val)
# print("Classification Report:\n %s:" % (metrics.classification_report(Y_val, predicted)))

results = model.predict(test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
submission = pd.read_csv('../input/sample_submission.csv')
submission["Label"] = results
submission.to_csv('./submission.csv', index=False)
# import pandas as pd
# import numpy as np
# import keras.layers.core as core
# import keras.layers.convolutional as conv
# import keras.models as models
# import keras.utils.np_utils as kutils

# # The competition datafiles are in the directory ../input
# # Read competition data files:
# train = pd.read_csv("../input/train.csv").values
# test  = pd.read_csv("../input/test.csv").values

# nb_epoch = 1 # Change to 100

# batch_size = 128
# img_rows, img_cols = 28, 28

# nb_filters_1 = 32 # 64
# nb_filters_2 = 64 # 128
# nb_filters_3 = 128 # 256
# nb_conv = 3

# trainX = train[:, 1:].reshape(train.shape[0], img_rows, img_cols, 1)
# trainX = trainX.astype(float)
# trainX /= 255.0

# trainY = kutils.to_categorical(train[:, 0])
# nb_classes = trainY.shape[1]

# cnn = models.Sequential()

# cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
# cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
# cnn.add(conv.MaxPooling2D(strides=(2,2)))

# cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
# cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
# cnn.add(conv.MaxPooling2D(strides=(2,2)))

# #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
# #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
# #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
# #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
# #cnn.add(conv.MaxPooling2D(strides=(2,2)))

# cnn.add(core.Flatten())
# cnn.add(core.Dropout(0.2))
# cnn.add(core.Dense(128, activation="relu")) # 4096
# cnn.add(core.Dense(nb_classes, activation="softmax"))

# cnn.summary()
# cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

# testX = test.reshape(test.shape[0], 28, 28, 1)
# testX = testX.astype(float)
# testX /= 255.0
# yPred = cnn.predict_classes(testX)
# print(yPred.shape)
# np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


