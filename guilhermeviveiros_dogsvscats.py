import os

from tensorflow.python.client import device_lib

import numpy as np

import shutil

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



!ls ../input/
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip

!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip

!ls ../working
!mkdir all_images
import os

import shutil



train_dir = '../working/train'

dest_dir = '../working/all_images'



counter = 0





for subdir, dirs, files in os.walk(train_dir):

    

    for file in files:

        full_path = os.path.join(subdir, file)

        shutil.copy(full_path, dest_dir)

        counter = counter + 1

        

print(counter)
subdirs, dirs, files = os.walk('../working/all_images').__next__()

m = len(files)





filenames = []

labels = np.zeros((m, 1))





images_dir = '../working/all_images'

filenames_counter = 0





for subdir, dirs, files in os.walk(train_dir):

    

    for file in files:

        filenames.append(file)

                                    

        if 'cat' in file: labels[filenames_counter, 0]  = 1;

        else : labels[filenames_counter, 0] = 0;

    

        filenames_counter = filenames_counter + 1

    

    

print(len(filenames))

print(labels.shape)
from sklearn.utils import shuffle



filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
from sklearn.model_selection import train_test_split



# Used this line as our filename array is not a numpy array.

filenames_shuffled_numpy = np.array(filenames_shuffled)



X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(

    filenames_shuffled_numpy, y_labels_shuffled, test_size=0.2, random_state=1)



X_val_filenames, X_test_filenames, y_val, y_test = train_test_split(

    X_val_filenames, y_val, test_size=0.5, random_state=1)



print(X_train_filenames.shape)

print(y_train.shape)          



print(X_val_filenames.shape)  

print(y_val.shape)            



print(X_val_filenames.shape)  

print(y_val.shape)

import keras

from PIL import Image



#this function receive keras.utils.Sequence to be compatible with the keras models in runtime

class My_Custom_Generator(keras.utils.Sequence) :

  

  def __init__(self, image_filenames, labels, batch_size) :

    self.image_filenames = image_filenames

    self.labels = labels

    self.batch_size = batch_size

    

  def __len__(self) :

    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  

  

  def __getitem__(self, idx) :

    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    

    return np.array([

            np.array(Image.open('../working/all_images/' + str(file_name)).resize((120,120)))

               for file_name in batch_x])/255.0, np.array(batch_y)
batch_size = 128



my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)

my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)
l = my_training_batch_generator.__getitem__(1)
print(l[0].shape)

#if its a gray scale image, add this shit -> cmap = plt.get_cmap('gray')

plt.imshow(l[0][11])
# from keras.losses import binary_crossentropy

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.losses import binary_crossentropy

from keras.layers import * 

from keras.models import Model, Sequential

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.utils import plot_model



import keras.backend as K

K.set_image_data_format('channels_last')
def build_model():

    model = Sequential()



    model.add(ZeroPadding2D((3, 3),input_shape=(120,120,3)))



    model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.5))





    model.add(Conv2D(filters = 512, kernel_size = (3,3), activation ='relu'))

    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters = 512, kernel_size = (3,3), activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization(axis=3))

    model.add(Dropout(0.5))



    model.add(Flatten())



    model.add(Dense(256, activation = "relu")) #Fully connected layer

    model.add(BatchNormalization())

    model.add(Dropout(0.5))





    model.add(Dense(60, activation = "relu")) #Fully connected layer

    model.add(BatchNormalization())

    model.add(Dropout(0.6))

    

    model.add(Dense(1, activation='sigmoid', name='fc'))



    return model



#build the model

model = build_model()



#adam as optimizer and binary cross entropy as metric

model.compile(optimizer = 'adam', loss = binary_crossentropy , metrics = ['accuracy'])



model.summary()

    

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.losses import binary_crossentropy



callbacks = [

        ModelCheckpoint(filepath="best_weights.hdf5", 

                               monitor = 'val_accuracy',

                               mode = 'max',

                               verbose=1,

                               period = 2,

                               save_best_only=True),

        EarlyStopping(monitor='val_accuracy',

                      mode = 'max',

                      patience = 5)

]



history = model.fit(my_training_batch_generator,

                   #steps_per_epoch = int(20000 // batch_size),

                   epochs = 10,

                   verbose = 2,

                   validation_data = my_validation_batch_generator,

                   #validation_steps = int(5000 // batch_size),

                   callbacks= callbacks,

                   shuffle = True

                   )
def show_history(history):

    print(history.history.keys())

    

    #Regarding to the accuracy and val_accuracy of the model

    plt.plot(history.history["accuracy"])

    plt.plot(history.history["val_accuracy"])

    plt.title("Accuracy of the model")

    plt.ylabel("Accuracy")

    plt.xlabel("Epoch")

    plt.show()

    

    #Regarding to the losses of the model

    plt.plot(history.history["loss"])

    plt.plot(history.history["val_loss"])

    plt.title("Loss of the model")

    plt.ylabel("Loss")

    plt.xlabel("Epoch")

    plt.show()
#print(history.history.keys())

show_history(history)
model.evaluate(X_test_filenames,y_test)
#model.summary()

plot_model(model)
# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
from keras.models import model_from_json



json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")

print("Loaded model from disk")
loaded_model.compile(optimizer = 'adam', loss = binary_crossentropy , metrics = ['accuracy'])

loaded_model.evaluate(y_test_set,y_result_set)