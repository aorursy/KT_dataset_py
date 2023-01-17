#Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
from sklearn.metrics import confusion_matrix


#Input data files
train = pd.read_csv ("../input/train.csv")
test = pd.read_csv ("../input/test.csv")

#Assign of values
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')

#Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0


#Reshape the images to 28x28x1 3D matrices 
#3 dimensions (height = 28px, width = 28px , canal = 1) for Keras needed

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)


#Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

#number of samples per batch
batch_size = 64 
#amount of recognizable digits (0-9) 
num_classes = 10
# number of iterations
epochs = 1 
input_shape = (28, 28, 1)

# convert to binary class matrices (One Hot Encoding)
# example: 2 -> [0,0,1,0,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, num_classes)

#split training set and validation set for fitting
#10% for the validation set and 90% is used to train the model
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
#Define the model
model = Sequential()
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
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

#Compile the model; measures the perfomance of the model on images
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

#data augmentation to prevent overfitting
genData = ImageDataGenerator(
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

model.summary()
# Method "fit", which trains the model
genData.fit(X_train)

# More iterations with 15 epochs
h = model.fit_generator(genData.flow(X_train, Y_train, batch_size=batch_size),
                               epochs = 15, validation_data = (X_val, Y_val),
                               verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                                , callbacks=[learning_rate_reduction],)
# Evaluate
final_loss, final_accuracy = model.evaluate(X_val, Y_val, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_accuracy))
# Get Prediction for test-data
results = model.predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series (results, name = "Label")

print (results)
# Get Image 15. stage
t = plt.imshow (X_test[15] [:,:,0])
# Submission
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv ("sample_submission.csv", index=False)
