#Import All Necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns #Data Visulation 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical #convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout,BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping







import keras 

from keras.datasets import mnist

import tensorflow as tf



print("Tensorflow version " + tf.__version__)





np.random.seed(123)
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
#check the file in the dicretory 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#load data from that dicretory 

train_data=pd.read_csv("../input/digit-recognizer/train.csv")

test_data=pd.read_csv("../input/digit-recognizer/test.csv")
#separate the independent and dependent variables (values of X and Y)



Y_train=train_data['label']



#drop "lable" column 

X_train=train_data.drop('label', axis=1)

(x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()



x_train1 = np.concatenate([x_train0, x_test0], axis=0)

y_train1 = np.concatenate([y_train0, y_test0], axis=0)



X_train_keras = x_train1.reshape(-1, 28*28)

Y_train_keras = y_train1

X_train = np.concatenate((X_train.values, X_train_keras))

Y_train = np.concatenate((Y_train, Y_train_keras))
print(X_train.shape)

print(Y_train.shape)
#Statistical summary for test data

print(test_data.shape)

test_data.head()
#Counts images for every digit

unique, counts = np.unique(Y_train, return_counts=True)

dict(zip(unique, counts))






#Diplay bar chart 

sns.set(context='notebook', style='darkgrid', palette='deep')

g = sns.countplot(Y_train)
#convert values to float

X_train = X_train.astype('float32')

Y_train = Y_train.astype('float32')

test_data=test_data.astype('float32')



# Normalize the data

X_train = X_train / 255.0

test_data = test_data / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 9 -> [0,0,0,0,0,0,0,0,0,1])

Y_train=to_categorical(Y_train, num_classes=10)



print(f"Label size {Y_train.shape}")

# Split the train and the validation set for the fitting



X_train, X_val, Y_train, Y_val=train_test_split(X_train, Y_train, test_size=0.10, random_state=44)



# 10% for Validation data, 90% for training data
#print the sizes of datasets

print("The size of X_train : {}\nThe size of Y_train : {}\nThe size of X_val   : {}\nThe size of Y_val   : {}\n"

      .format(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape))



#Conver X_train to shape (num_images, img_rows, img_cols) for plotting 

X_train_temp = X_train.reshape(X_train.shape[0], 28, 28)



fig, axis = plt.subplots(3, 4, figsize=(20, 10))

for i, ax in enumerate(axis.flat):

    ax.imshow(X_train_temp[i], cmap='binary')

    digit = Y_train[i].argmax()

    ax.set(title = f"Real Number is {digit}");
model= Sequential()



model.add(Conv2D(input_shape=(28,28,1), filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.10))



model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.10))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

model.add(BatchNormalization())







model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))









model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(10, activation='softmax'))



model.summary()
from keras.utils import plot_model

plot_model(model, to_file='model_chart.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image("model_chart.png")
#Define the optimizer

optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#cmpile the model

model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# Set a learning rate annealer

learning_rate_redcuing=ReduceLROnPlateau(monitor='val_accuracy', 

                                         patience=2,

                                         verbose=1,

                                         factor=0.5,

                                         min_lr=0.00001)
#stops training when accuracy do not improved

#earlystopper = EarlyStopping(monitor='val_accuracy', min_delta=0,

               #              patience=6, verbose=1, mode='auto')
epochs = 50 # 

batch_size = 64
#Do data augmentation to prevent overfitting



imagegen=ImageDataGenerator(

                            featurewise_center=False, #set input mean to 0 over the dataset

                            samplewise_center=False, #set each sample mean to 0

                            featurewise_std_normalization=False, #divide inputs by std of the dataset

                            samplewise_std_normalization=False, #divide each input by its std

                            zca_whitening=False, #apply ZCA whitening

                            rotation_range=10, #randomly rotate images in the range (degrees, 0 to 180)

                            zoom_range=0.1, #randomly zoom image 

                            width_shift_range=0.1, #randomly shift images horizontally (fraction of total width)

                            height_shift_range=0.1, #randomly shift images vertically (fraction of total height)

                            horizontal_flip=False, #randomly flip images

                            vertical_flip=False)

    

imagegen.fit(X_train)
#Training data (Fit the model)



history=model.fit_generator(imagegen.flow(X_train, Y_train,batch_size=batch_size),

                                          epochs=epochs,

                                          validation_data=(X_val, Y_val),

                                          verbose=2,

                                          steps_per_epoch=X_train.shape[0] // batch_size,

                                          callbacks=[learning_rate_redcuing])

                                          

#Save the model

model.save("MNIST_CNN_Model.h5")

model.save_weights("MNIST_CNN_Model_weights.h5")
# Plot the loss and accuracy curves for training and validation 



fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training Loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training Accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation Accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Predict the values from the validation dataset

Y_pred=model.predict(X_val)



#Convert predictions classes to one hot vectors

Y_pred_classes=np.argmax(Y_pred, axis=1)



# Convert validation observations to one hot vectors

Y_true=np.argmax(Y_val, axis=1)



# compute the confusion matrix

confusion_mtx=confusion_matrix(Y_true, Y_pred_classes)



# plot the confusion matrix

plt.figure(figsize=(10, 10)) #The size of plot chart

conf_plot=sns.heatmap(confusion_mtx, annot=True, fmt='d', linewidths=.1, linecolor='black', cmap="YlGnBu", square=True)



#set title and labels

conf_plot.set(xlabel="Predicted label", ylabel = 'True label', title='Confusion Matrix')
#Display some error results 



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
y_pred = model.predict(X_val)

X_test_temp = X_val.reshape(X_val.shape[0], 28, 28)



fig, axis = plt.subplots(4, 4, figsize=(12, 14))

for i, ax in enumerate(axis.flat):

    ax.imshow(X_test_temp[i], cmap='binary')

    ax.set(title = f"Real Number is {Y_val[i].argmax()}\nPredict Number is {y_pred[i].argmax()}");
# predict results

results = model.predict(test_data)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("sample_submission.csv",index=False)



submission.head()