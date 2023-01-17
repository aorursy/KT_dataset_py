import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam # I believe this is better optimizer for our case
from keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from keras.utils.vis_utils import plot_model
import scipy
from sklearn.model_selection import train_test_split # to split our train data into train and validation sets
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(13) # My lucky number
num_classes = 10 # We have 10 digits to identify
batch_size = 128 # Handle 128 pictures at each round
epochs = 700 
img_rows, img_cols = 28, 28 # Image dimensions 28 pixels in height&width
input_shape = (img_rows, img_cols,1) # We'll use this while building layers
# Load some date to rock'n roll
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Drop the label from the data and move it to real label part
y_train = train["label"]
x_train = train.drop(labels = ["label"],axis = 1 )
# Normalize both sets
x_train /= 255
test /= 255
print(x_train.shape[0], 'train samples')
print(test.shape[0], 'test samples')
# Images should be in shape of height,width and color channel so it will be 28x28x1
x_train = x_train.values.reshape(-1,img_rows,img_cols,1).astype('float32')
test = test.values.reshape(-1,img_rows,img_cols,1).astype('float32')
# Class vectors needs to be binary so we use "to_catogorical" function of keras utilities for one-hot-encoding
y_train = keras.utils.to_categorical(y_train, num_classes = num_classes)
# Lets split our train set into train and validation test sets with my lucky number 13 :)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1)
def model_cnn(input_shape=input_shape, num_classes=num_classes):   
    model = Sequential()

    # Add convolutional layer consisting of 32 filters and shape of 3x3 with ReLU activation
    # We want to preserve more information for following layers so we use padding
    # 'Same' padding tries to pad evenly left and right, 
    # but if the amount of columns to be added is odd, it will add the extra column to the right
    model.add(Conv2D(32, kernel_size = (3,3), activation='relu', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = (3,3), activation='relu'))
    model.add(BatchNormalization())

    # Add convolutional layer consisting of 32 filters and shape of 5x5 with ReLU activation
    # We give strides=2 for space between each sample on the pixel grid
    model.add(Conv2D(32, kernel_size = (5,5), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    # Dropping %40 of neurons
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, kernel_size = (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = (5,5), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    # To be able to merge into fully connected layer we have to flatten
    model.add(Flatten())
    model.add(Dropout(0.4))
    # Lets add softmax activated neurons as much as number of classes
    model.add(Dense(num_classes, activation = "softmax"))
    # Compile the model with loss and metrics
    model.compile(optimizer =  Adam() , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    return model
def LeNet5(input_shape=input_shape,num_classes=num_classes):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer =  Adam() , loss = "categorical_crossentropy", metrics=["accuracy"])
    return model
print("My Custom CNN Network:")
plot_model(model_cnn(), to_file='custom-cnn.png', show_shapes=True, show_layer_names=True)
print("Master Yann LeCun's LeNet-5 Network:")
plot_model(LeNet5(), to_file='lenet-5.png', show_shapes=True, show_layer_names=True)
model = []
model.append(model_cnn())
model.append(LeNet5())
# Generate batches of tensor image data with real-time data augmentation more detail: https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)
# Start multiple model training with the batch size
models = []
for i in range(len(model)):
    model[i].fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                                        epochs = epochs, steps_per_epoch=x_train.shape[0] // batch_size,
                                        validation_data = (x_test,y_test), 
                                        callbacks=[ReduceLROnPlateau(monitor='loss', patience=3, factor=0.1)], 
                                        verbose=2)
    models.append(model[i])
# Predict labels with models
labels = []
for m in models:
    predicts = np.argmax(m.predict(test), axis=1)
    labels.append(predicts)
    
# Ensemble with voting
labels = np.array(labels)
labels = np.transpose(labels, (1, 0))
labels = scipy.stats.mode(labels, axis=-1)[0]
labels = np.squeeze(labels)
# Dump predictions into submission file
pd.DataFrame({'ImageId' : np.arange(1, predicts.shape[0] + 1), 'Label' : labels }).to_csv('submission.csv', index=False)