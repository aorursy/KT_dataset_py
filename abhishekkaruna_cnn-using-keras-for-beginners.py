# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from os import getcwd, chdir

import pathlib

path = '/kaggle/input/'

chdir(path)

print(getcwd())



#Current Directory
data_root = pathlib.Path(getcwd()+'/intel-image-classification/seg_train/seg_train')

print ()
import random



# create list of all image paths under data_root in 'pathlib.PosixPath' type

all_image_paths = list(data_root.glob('*/*'))

print(type(all_image_paths[0]))

print(all_image_paths[0])

# convert path list from 'pathlib.PosixPath' type to string

all_image_paths = [str(path) for path in all_image_paths]

print(type(all_image_paths[0]))

print(all_image_paths[0])

# shuffle up image paths. This is mostly to view a variety of images during data

# investigation. Final dataset will be shuffled again during training

random.seed(0)

random.shuffle(all_image_paths) 



image_count = len(all_image_paths)

image_count
all_image_paths[:10] # Verify paths are as expected
import IPython.display as display

for n in range(3):

  image_path = random.choice(all_image_paths)

  display.display(display.Image(image_path))

  print(image_path)

  print()
# Importing the Keras libraries and packages



from keras.models import Sequential   

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras import regularizers

from keras import optimizers as Optimizer

from keras.layers import Dropout





from keras.models import load_model

import keras.backend as K

import numpy as np

from sklearn.metrics import confusion_matrix

import numpy as np

from matplotlib import image

import matplotlib.pyplot as plt



#Sequential function is used to initialize the neural networks [CNN]

model = Sequential()
#Three Convolution Layers are built below



#Convolution layer is built

model.add(Conv2D(128, (5, 5), activation='relu',input_shape=(150, 150, 3),padding='same',kernel_regularizer=regularizers.l2(0.0025)))

#Input Shape is 150 * 150 as the dimension is same and the thrid layer is 3 as this is a RGB scenario

#Feature maps: Input Image * Featue detector [128 in this case]

#Dimension of feature detecor taken: No of Rows and Coloumn in Feature Detectors [5*5]

#Activation Layer: Rectfier

#Rectifier is applied to increase nonlinearity to our network

#Padding is used instead of Strdie and Regularization is also used 





model.add(MaxPooling2D((3, 3)))

#Max Pooling helps us to preserve major features despite reducing the 75% of content

#Max Pooling reduces the no of parameters and helps to prevent over fititng

#Max pooling of 3*3 is used here



model.add(Dropout(0.5))

#Dropout Regularization is used to reduce overfitting if needed

#In each iteration, some neurons from ANN are randomly disabled to prevent any form of interdepedency

#Two more layers are built similarly

model.add(Conv2D(128, (5, 5), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0025)))

model.add(MaxPooling2D((5, 5)))

model.add(Dropout(0.3))

model.add(Conv2D(256, (5, 5), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0025)))

model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

#All the ouputs of convolution layer is converted to 1D array by flattening

#This Array will be feaded to hidden layer as inputs
#Fully connected Dense layer 

model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.002)))

#Units: 256

#Usually taken on the basis of Average of Input and Output. Output is 6 in our case 



model.add(Dropout(0.5))

model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.002)))

model.add(Dense(6, activation='softmax'))

#Last layer activation is softmax as the output categories is more than two
model.compile(loss='categorical_crossentropy',

              optimizer=Optimizer.Adam(lr=0.0008),

              metrics=['acc'])



#Optimizier: Adam

#Stochastic Gradient Descent is used

#Loss: Categorical cross entropy as the depedent variable is more than two

#Regularization is used
model.summary()
#Data Augmentation is done on the training set below



#Image Augmentation is used to increase the variations of images by fliping, rotating, shifting and etc





from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)





#train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,)



train_dir = 'intel-image-classification/seg_train/seg_train'

train_validation = 'intel-image-classification/seg_test/seg_test'



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(150, 150),

    batch_size=100,

    class_mode='categorical')



# Note that the validation data shouldn't be augmented!

test_datagen = ImageDataGenerator(rescale=1./255)



validation_generator = test_datagen.flow_from_directory(

        train_validation,

        target_size=(150, 150),

        batch_size=100,

        class_mode='categorical')



#Until now the neural network is built without any connection. The fit function helps to connect with the training set

#Batch size is 100

#Epochs: How many time the model must run

#Epochs: 100 in our case 



history = model.fit_generator(

      train_generator,

      steps_per_epoch=14000/100,

      epochs=100,

      validation_data=validation_generator,

      validation_steps=3000/100)

#Plotting Training and Validation Accuracy and Losses



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)





import matplotlib.pyplot as plt



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()

from os import getcwd, chdir

import pathlib





label_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

train_size = sum([len(list(pathlib.Path(train_dir,label).iterdir())) for label in label_names])

val_size = sum([len(list(pathlib.Path(train_validation,label).iterdir())) for label in label_names])


from keras.models import load_model

import keras.backend as K

import numpy as np

from sklearn.metrics import confusion_matrix

import numpy as np

from matplotlib import image

import matplotlib.pyplot as plt





#val_size = sum([len(list(Path(validation_dir,label).iterdir())) for label in label_names])



steps_per_epoch = train_size / 100

validation_steps = val_size / 100

print(steps_per_epoch)

print(validation_steps)









test_labels = validation_generator.labels

test_images = np.array([image.imread(fpath) for fpath in validation_generator.filepaths])

predictions = model.predict_generator(validation_generator, steps=validation_steps)

pred_labels = np.argmax(predictions, axis = 1)
#Confusion Matrix



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

 



#This function prints and plots the confusion matrix.

#Normalization can be applied by setting `normalize=True

        

    

    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized")

    #else:

        #print('Not Normalized')



    #print(cm)

    fig, ax = plt.subplots()

    #fig = plt.figure(figsize=(15,5))

    #plt.subplot(1,2,1)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

            

    fig.tight_layout()

    return ax
# Note the mountain/glacier, building/street, sea/glacier confusion. 

plot_confusion_matrix(test_labels, pred_labels, classes=label_names, title='Not Normalized')

plot_confusion_matrix(test_labels, pred_labels, classes=label_names, normalize=True, title='Normalized')