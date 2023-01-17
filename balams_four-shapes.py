#Load Basic libraries
import numpy as np
import pandas as pd
import os
import cv2

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Train Test Split
from sklearn.model_selection import train_test_split

#Model
import keras
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Flatten, Activation, Dropout
#Load Dataset
PATH = '../input/shapes'
img_size = 64
shapes = ['circle', 'square','triangle', 'star']
labels = []
dataset = []

for shape in shapes:
    print('Getting Data for :', shape)
    for img in os.listdir(os.path.join(PATH, shape)):
        image = cv2.imread(os.path.join(PATH,shape,img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(img_size,img_size))
        dataset.append(image)
        #Add Label list
        labels.append(shapes.index(shape))
    
print('\nDataset Size :', len(dataset))
print('Image Shape :', dataset[0].shape)
print('Label size :', len(labels))
print('Count of Circle Images :', labels.count(shapes.index('circle')))
print('Count of Square Images :', labels.count(shapes.index('square')))
print('Count of Triangle Images :', labels.count(shapes.index('triangle')))
print('Count of Star Images :', labels.count(shapes.index('star')))

#View Random Images from dataset
index = np.random.randint(0, len(dataset), size = 20)

plt.figure(figsize = (15,5))
for i, ind in enumerate(index,1):
    img = dataset[ind]
    lab = labels[ind]
    plt.subplot(2,10,i)
    plt.title(shapes[lab])
    plt.axis('off')
    plt.imshow(img, cmap = 'gray')
dataset_np = np.array(dataset).reshape(-1, 64, 64, 1)
dataset_np.shape
#Normalization Image
dataset_np = dataset_np.astype('float32')/255.0
dataset_np
#Split train and test set
x_train, x_test, y_train, y_test = train_test_split(dataset_np, labels, test_size = 0.3, random_state = 7)

print('x_train size is :', x_train.shape)
print('x_test size is :', x_test.shape)
print('y_train size is :', len(y_train))
print('y_test size is :', len(y_test))
#Model Creation
model = Sequential()
model.add(Conv2D(32, 10, input_shape = x_train[1].shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
trained_model = model.fit(x_train, y_train, validation_split= 0.1, batch_size = 10, epochs = 10, verbose = 1)
pred = model.predict_classes(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred)
plt.figure(figsize = (15,7))
plt.subplot(2,1,1)
sns.lineplot(x = np.arange(0, 10), y = trained_model.history['acc'], label="Max Pool Train Acc")
sns.lineplot(x = np.arange(0, 10), y = trained_model.history['val_acc'],label="Max Pool Test Acc" )

plt.subplot(2,1,2)
sns.lineplot(x = np.arange(0, 10), y = trained_model.history['loss'], label="Max Pool Train Loss")
sns.lineplot(x = np.arange(0, 10), y = trained_model.history['val_loss'],label="Max Pool Test Loss" )

plt.show()