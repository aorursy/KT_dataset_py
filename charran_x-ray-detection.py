import glob

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import os

from sklearn .preprocessing import LabelBinarizer

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from sklearn.metrics import classification_report,confusion_matrix

import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

DATASET_DIR = "../input/covid-19-x-ray-10000-images/dataset"
normal_images = []

for img_path in glob.glob(DATASET_DIR + '/normal/*'):

    normal_images.append(mpimg.imread(img_path))

    

fig = plt.figure()

fig.suptitle('normal')

plt.imshow(normal_images[0], cmap='gray') 



covid_images = []

for img_path in glob.glob(DATASET_DIR + '/covid/*'):

    covid_images.append(mpimg.imread(img_path))



fig = plt.figure()

fig.suptitle('covid')

plt.imshow(covid_images[0], cmap='gray') 
IMG_W = 150

IMG_H = 150

CHANNELS = 3



INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)

NB_CLASSES = 2

EPOCHS = 41

BATCH_SIZE = 6
model = Sequential()

model.add(Conv2D(32 , (3,3) , padding = 'same' , strides = 1 , activation = 'relu' , input_shape = INPUT_SHAPE))

model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , padding = 'same' , strides = 1 , activation = 'relu'))

model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , padding = 'same' , strides = 1 , activation = 'relu'))

model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , padding = 'same' , strides = 1 , activation = 'relu'))

model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(256 , (3,3) , padding = 'same' , strides = 1 , activation = 'relu'))

model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())

model.add(Dense(units = 64 , activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(units = 1 , activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model.summary()
datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False, 

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=True,  

        vertical_flip=False,  

        rescale = 1./255,

        validation_split = 0.3) 

train_generator = datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_H, IMG_W),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='training')

validation_generator = datagen.flow_from_directory(

    DATASET_DIR, 

    target_size=(IMG_H, IMG_W),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    shuffle= False,

    subset='validation')

history = model.fit(train_generator,validation_data = validation_generator,epochs = EPOCHS)
print("Training Accuracy of the model is - " , model.evaluate(train_generator)[1]*100 , "%")

print("Validation Accuracy of the model is - " , model.evaluate(validation_generator)[1]*100 , "%")
epochs = [i for i in range(41)]

fig , ax = plt.subplots(1,2)

train_acc = history.history['accuracy']

train_loss = history.history['loss']

test_acc = history.history['val_accuracy']

test_loss = history.history['val_loss']

fig.set_size_inches(20,10)



ax[0].plot(epochs , train_acc , 'g' , label = 'Training Accuracy')

ax[0].plot(epochs , test_acc , 'r' , label = 'Validation Accuracy')

ax[0].set_title('Training & Validation Accuracy')

ax[0].legend()

ax[0].set_xlabel("Epochs")

ax[0].set_ylabel("Accuracy")



ax[1].plot(epochs , train_loss , 'g' , label = 'Training Loss')

ax[1].plot(epochs , test_loss , 'r' , label = 'Validation Loss')

ax[1].set_title('Training & Validation Loss')

ax[1].legend()

ax[1].set_xlabel("Epochs")

ax[1].set_ylabel("Loss")

plt.show()
true_class = validation_generator.classes
pred_class = model.predict(validation_generator)

pred_class = np.around(pred_class , decimals = 0)
print(classification_report(true_class, pred_class, target_names = ['0','1']))
cm = confusion_matrix(true_class,pred_class)

cm
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
plt.figure(figsize = (10,10))

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , annot = True, fmt='')
correct = np.nonzero(pred_class == true_class)[0]
pred_class = pred_class.astype(int)
i = 0

for c in correct[:6]:

    plt.subplot(3,2,i+1)

    plt.imshow(validation_generator[0][0][c].reshape(150,150,3))

    plt.title("Predicted Class {},Actual Class {}".format(pred_class.reshape(1,-1)[0][c], true_class[c]))

    plt.tight_layout()

    i += 1