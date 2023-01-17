# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing |Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

import glob

import random

from sklearn.utils import shuffle

import seaborn as sns

import keras

# Any results you write to the current directory are saved as output.
def loadImages(root_dir,mode='train',labels_dict={'street':0,'sea':1,'glacier':2,'mountain':3,'buildings':4,'forest':5}):

    image_sizes=[]

    images=[]

    scene_type=[]

    if mode=='train':

        # Read the folders inside the root directory and the label is the name of the sub-directory

        for labels in os.listdir(root_dir):

            label=labels_dict[labels]

            for image_file in os.listdir(root_dir+"/"+labels):

                image = cv2.imread(root_dir+labels+r'/'+image_file)

                image_sizes.append(image.shape)

                ## There are images of multiple sizes in the data. Let us resize them

                

                image=cv2.resize(image,(150,150))

                images.append(image)

                scene_type.append(label)

        ### We need to shuffle the data set

        

        images,scenes= shuffle(images,scene_type,random_state=1234)

        

        ### Convert the list to numpy array 

        images=np.array(images)

        scenes=np.array(scenes)

        

        return images,scenes

    

                
images,labels=loadImages("../input/seg_train/seg_train/")
print("Shape of Images  in Training Data",images.shape)

print("Shape of Labels in Training Data",labels.shape)
sns.countplot(labels)
labels_dict={'street':0,'sea':1,'glacier':2,'mountain':3,'buildings':4,'forest':5}

inverse_labels={value:key for key,value in labels_dict.items()}

num_random=5



f,ax = plt.subplots(5,5) 

f.subplots_adjust(0,0,3,3)

for i in range(0,5,1):

    for j in range(0,num_random,1):

        

        rnd_number = random.randint(0,len(images))

        ax[i,j].imshow(images[rnd_number])

        ax[i,j].set_title(inverse_labels[labels[rnd_number]])

        ax[i,j].axis('off')
### In keras, data generators are used to allow training of data in batches. Keras, by default offers an Image DataGenerator, but many a times you will need to customise the data generation. So we will build our own data generator
TRAIN_DATASET_PATH="../input/seg_train/seg_train/"

import keras

from keras import models as Models

from keras import layers as Layers

from keras import optimizers as Optimizers

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    

    def __init__(self, mode='train', ablation=None, image_cls={'street':0,'sea':1,'glacier':2,'mountain':3,'buildings':4,'forest':5}, 

                 batch_size=32, dim=(150, 150), n_channels=3, shuffle=True,train_test_split=0.8):

        """

        Initialise the data generator

        """

        self.dim = dim

        self.batch_size = batch_size

        self.labels = {}

        self.list_IDs = []

        

        # glob through directory of each class 

        label_class=[key for key,val in image_cls.items()]

        for i, cls in enumerate(label_class):

            paths = glob.glob(os.path.join(TRAIN_DATASET_PATH, cls, '*'))

            brk_point = int(len(paths)*train_test_split) #Divide the data into 80:20 - training and validation set

            if mode == 'train':

                paths = paths[:brk_point]

            else:

                paths = paths[brk_point:]

            if ablation is not None:

                paths = paths[:ablation]

            self.list_IDs += paths

            self.labels.update({p:i for p in paths})

            

        self.n_channels = n_channels

        self.n_classes = len(label_class)

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

        y = np.empty((self.batch_size), dtype=int)

        

       



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            img = cv2.imread(ID)

            img = img/255

            ## Resize the image

            img=cv2.resize(img,self.dim)

            X[i,] = img

          

            # Store class

            y[i] = self.labels[ID]

        

        

        return X, y


model = Models.Sequential()



model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Flatten())

model.add(Layers.Dense(180,activation='relu'))

model.add(Layers.Dense(100,activation='relu'))

model.add(Layers.Dense(50,activation='relu'))

model.add(Layers.Dropout(rate=0.5))

model.add(Layers.Dense(6,activation='softmax'))





model.compile(optimizer=Optimizers.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])



training_generator=DataGenerator('train',train_test_split=0.7)

validation_generator=DataGenerator('val')

history=model.fit_generator(generator=training_generator,

                    validation_data=validation_generator,

                    epochs=35)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.save("CNN_Model_1.h5")
img=images[0]

plt.imshow(img)

img = img/255

## Resize the image

img=cv2.resize(img,(150,150))

img_tensor=np.empty((1,150,150,3))

img_tensor[0,] = img
from keras import models
model.summary()
layer_outputs=[layer.output for layer in model.layers] 
activation_model = Models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*10,col_size*10))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')

            activation_index += 1

            
display_activation(activations,8,8,0)