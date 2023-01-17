#importing all required libraries
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
directory = '../input/intel-image-classification/seg_train/seg_train/'
#creating empty arrays to store the images and their corresponding labels
images = []
labels = []  # 0-Building, 1-Forest, 2-Glacier, 3-Mountain, 4-Sea, 5-Street
lbl = 0
#extracting data and storing in arrays
for label in os.listdir(directory):
    if label == 'buildings':
        lbl = 0
    elif label == 'forest':
        lbl = 1
    elif label == 'glacier':
        lbl = 2
    elif label == 'mountain':
        lbl = 3
    elif label == 'sea':
        lbl = 4
    elif label == 'street':
        lbl = 5

    for image_file in os.listdir(directory+label):
        #reading the images using cv2
        image = cv2.imread(directory+label+r'/'+image_file)
        #resizing the images to 150X150 pixels
        image = cv2.resize(image,(150,150)) 
        #adding image to array
        images.append(image)
        #adding label to array
        labels.append(lbl)

#shuffling the data for training
shuffle(images,labels,random_state=817328462)

#converting the arrays to numpy arrays to feed to the model
images = np.array(images)
labels = np.array(labels)
#creating model
model = Models.Sequential()

#specifying input shape in first layer
model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3))) 
model.add(Layers.Conv2D(170,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(160,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(120,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))

#flattening for adding dense layers
model.add(Layers.Flatten()) 
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(140,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))

model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(6,activation='softmax')) #last layer - output layer

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

trained = model.fit(images,labels,epochs=30,validation_split=0.30)