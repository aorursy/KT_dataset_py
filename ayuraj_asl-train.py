import os

import cv2

import numpy as np

import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')



%matplotlib inline



import keras

from keras.models import Sequential

from keras import regularizers

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

from keras.utils import to_categorical





print(os.listdir("../input"))
path = "../input/asl_dataset (2)/asl_dataset"
dirs = os.listdir(path)

unique_labels = dirs
label_dict = {

    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,

    '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16,

    'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24,

    'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32,

    'x': 33, 'y': 34, 'z': 35,

}
size = 200,200



def load_dataset(path):

    image_data = list()

    labels = list()

    dirs = os.listdir(path)

    

    for Dir in dirs:

        print("Loading images for: ", Dir)

        path_img = path+'/'+Dir

        images = os.listdir(path_img)

        for image in images:

            temp_img = cv2.imread(path_img+'/'+image)

            temp_img = cv2.resize(temp_img, size)

            image_data.append(temp_img)

            labels.append(label_dict[Dir])

            

    return image_data, labels

        
images, labels = load_dataset(path)
images = np.array(images)

images = images.astype('float32')/255.0



labels = to_categorical(labels)
print("Shape of image data: ", images.shape)

print("shape of labels: ", labels.shape)
def create_model():

    

    model = Sequential()

    

    model.add(Conv2D(16, kernel_size = [7,7], padding = 'valid', activation = 'relu', input_shape = (200,200,3)))

    model.add(Conv2D(32, kernel_size = [5,5], padding = 'valid', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [2,2]))

    

    model.add(Conv2D(64, kernel_size = [5,5], padding = 'valid', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(Conv2D(64, kernel_size = [5,5], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))

    model.add(Dense(36, activation = 'softmax'))

    

    print("MODEL CREATED")

    return model
model = create_model()
model.summary()
model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
model_hist = model.fit(images, labels, batch_size = 32, epochs = 25, validation_split = 0.01)
def plot_accuracy(y):

    if(y == True):

        plt.plot(model_hist.history['acc'])

        plt.plot(model_hist.history['val_acc'])

        plt.legend(['train', 'test'], loc='lower right')

        plt.title('accuracy plot - train vs test')

        plt.xlabel('epoch')

        plt.ylabel('accuracy')

        plt.show()

    else:

        pass

    return



def plot_loss(y):

    if(y == True):

        plt.plot(model_hist.history['loss'])

        plt.plot(model_hist.history['val_loss'])

        plt.legend(['training loss', 'validation loss'], loc = 'upper right')

        plt.title('loss plot - training vs vaidation')

        plt.xlabel('epoch')

        plt.ylabel('loss')

        plt.show()

    else:

        pass

    return





plot_accuracy(True)

plot_loss(True)
model.save('asl_model_better.h5')