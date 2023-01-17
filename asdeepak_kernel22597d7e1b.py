# This Program demosnstrates Handwritten Digit Classification (MNIST Dataset) Using Generator function and 
# Data Generator Class. Here CNN model is developed. 
from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob
import random
from skimage import io
import matplotlib.pyplot as plt
% matplotlib inline
from keras_preprocessing.image import ImageDataGenerator
mypath = '/Users/Deepu/Data Science/Gowri/mnistasjpg/trainingSet/trainingSet'
digit_cls = ['0', '1','2','3','4','5','6','7','8','9']
num_classes = len(digit_cls)
#df=pd.read_csv("/content/drive/My Drive/mnist@images/trainingSet/trainingSet/trainingset.csv")
#df
train_path = glob.glob(join(mypath, digit_cls[0],'*'))
len(train_path)
rand_index = random.randint(0, len(train_path))
image = io.imread(train_path[rand_index])
plt.imshow(image,cmap='gray')
print(image.shape[0]) 
input_shape = (28,28,1)

labels = {}
list_IDs = []
for i, cls in enumerate(digit_cls):
        paths = glob.glob(join(mypath, cls, '*'))
        list_IDs += paths
        labels.update({p:i for p in paths})
list_IDs
random.shuffle(list_IDs)
list_IDs
Paths_df = pd.DataFrame(list(labels.items()), index=np.arange(0,len(labels)),columns=['Path', 'Label'])
Paths_df

Paths_df.to_csv("folders_list.csv", index=False, header=True)
def generator(folder_list, batch_size):
    
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)//batch_size # calculate the number of batches
        pending_Data=len(folder_list) % batch_size
        
        for batch in range(num_batches): # we iterate over the number of batches
            #print(batch)
            x = np.empty([batch_size,28,28])
            y = np.empty(batch_size, dtype=int)
            for i in range(batch_size): # iterate over the batch_siz
                ID = t[i + (batch*batch_size)]
                img = io.imread(ID).astype(np.float32)
                img = img/255
                x[i,] = img
                y[i] = labels[ID]
                    
            x = np.reshape(x,(x.shape[0],28,28,1))
            y = keras.utils.to_categorical(y, num_classes=num_classes)
            yield x, y #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches
        if pending_Data>0:
            #print("pending_Data"+str(pending_Data))
            batch_size=pending_Data
            #print(batch)
            x = np.empty([batch_size,28,28])
            y = np.empty(batch_size, dtype=int)
            for i in range(batch_size): # iterate over the batch_siz

                img = io.imread( t[i + (batch*batch_size)]).astype(np.float32)
                img = img/255
                x[i,] = img
                y[i] = labels[ID]
                    
            x = np.reshape(x,(x.shape[0],28,28,1))
            y = keras.utils.to_categorical(y, num_classes=num_classes)
            yield x, y #you yield the batch_data and the batch_labels, remember what does yield do


data_size = len(list_IDs)
train_path = list_IDs[:int(0.8*data_size)]
val_path = list_IDs[int(0.8*data_size):]
test_path = list_IDs[int(0.95*data_size):]
data_size

input_shape = (28,28,1)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# model summary
model.summary()
# usual cross entropy loss
# choose any optimiser such as adam, rmsprop etc
# metric is accuracy
'''model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])'''
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

batch_size =32
num_train_sequences = len(train_path)
num_val_sequences = len(val_path)
num_test_sequences = len(test_path)
if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1

if (num_test_sequences%batch_size) == 0:
    test_steps = int(num_test_sequences/batch_size)
else:
    test_steps = (num_test_sequences//batch_size) + 1
train_generator = generator(train_path, batch_size)
val_generator = generator(val_path, batch_size)
test_generator = generator(test_path,batch_size)
history=model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=3, verbose=1, 
                            validation_data=val_generator,validation_steps=validation_steps, class_weight=None,
                            workers=6, initial_epoch=0, use_multiprocessing=True)
model.evaluate_generator(val_generator, steps=test_steps)
df = pd.read_csv('test.csv')
x_predict = df.values
x_predict = np.reshape(x_predict,(x_predict.shape[0],28,28,1))
df = pd.read_csv('test.csv')
x_predict = df.values
x_predict = np.reshape(x_predict,(x_predict.shape[0],28,28,1))
y_predcit = model.predict_class(x_predict)
#y =np.argmax( y_predcit,axis=1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(y)+1)),"Label": y})
submissions.to_csv("Submission_Deepak.csv", index=False, header=True)
y =np.argmax( y_predcit,axis=1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(y)+1)),
                         "Label": y})
submissions.to_csv("Submission_Deepak.csv", index=False, header=True)
 

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(28,28), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)
        
        x = np.empty([self.batch_size,28,28])
        y = np.empty(self.batch_size, dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = io.imread(ID)
            img = img/255
            x[i,] = img
            y[i] = labels[ID]

        X= np.reshape(x,(x.shape[0],28,28,1))
        Y = keras.utils.to_categorical(y, num_classes=num_classes)
        
        return X, Y
training_generator = DataGenerator(train_path, labels)
validation_generator = DataGenerator(val_path, labels)



# fit: this will fit the net on 'ablation' samples, only 1 epoch
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=30,use_multiprocessing=True,
                    workers=6)
model.evaluate_generator(val_generator, steps=test_steps)
y_predcit = model.predict(x_predict)
y =np.argmax( y_predcit,axis=1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(y)+1)),"Label": y})
submissions.to_csv("Submission_Deepak.csv", index=False, header=True)






