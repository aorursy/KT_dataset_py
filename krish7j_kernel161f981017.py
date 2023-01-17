# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input/chest_xray/chest_xray/"))

from glob import glob

from PIL import Image

import matplotlib.pyplot as plt

import cv2

import fnmatch

import keras

from time import sleep

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation

from keras.optimizers import RMSprop,Adam

from tensorflow.keras.callbacks import EarlyStopping

from keras import backend as k
print(os.listdir("../input/chest_xray/chest_xray/val/PNEUMONIA"))
!ls
import seaborn as sns

sns.barplot(x=["Normal", "Pneumonia"], y=[len(train_n), len(train_p)])
imagePatches = glob('../input/chest_xray/chest_xray/**/**/*.jpeg', recursive=True)

print(len(imagePatches))
pattern_normal = '*NORMAL*'

pattern_bacteria = '*_bacteria_*'

pattern_virus = '*_virus_*'



normal = fnmatch.filter(imagePatches, pattern_normal)

bacteria = fnmatch.filter(imagePatches, pattern_bacteria)

virus = fnmatch.filter(imagePatches, pattern_virus)

x = []

y = []

for img in imagePatches:

    full_size_image = cv2.imread(img)

    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    x.append(im)

    if img in normal:

        y.append(0)

    elif img in bacteria:

        y.append(1)

    elif img in virus:

        y.append(1)

    else:

        #break

        print('no class')

x = np.array(x)

y = np.array(y)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101)

y_train = to_categorical(y_train, num_classes = 2)

y_valid = to_categorical(y_valid, num_classes = 2)

del x, y

file_loc = "../input/chest_xray/chest_xray/"

train_n = os.listdir(file_loc + "train/NORMAL/")

train_p = os.listdir(file_loc + "train/PNEUMONIA/")

fig, axarr = plt.subplots(3, 2, figsize=(16, 16))

axarr[0][0].set_title("Normal Sample Cases")

axarr[0][1].set_title("Pneumonia Sample Cases")

for i in range(3):

    axarr[i][0].imshow(cv2.imread(file_loc + "train/NORMAL/" + train_n[i]))

    axarr[i][0].axis("off")

    axarr[i][1].imshow(cv2.imread(file_loc + "train/PNEUMONIA/" + train_p[i]))

    axarr[i][1].axis("off")

import keras

from keras.models import Sequential,Input,Model

from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU 



model = Sequential()



model.add(Conv2D(32,(7,7),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())





model.add(Conv2D(64,(5,5),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())





model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())





model.add(GlobalAveragePooling2D())

model.add(Dense(1000, activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

#model.summary()
from keras.models import Sequential

from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=250, facecolor='w', edgecolor='k')



plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

img = cv2.imread("model_plot.png",0)

plt.imshow(img,cmap = 'gray',interpolation = 'bicubic')
from keras.callbacks import ModelCheckpoint

mcp = ModelCheckpoint(filepath='model.h5',monitor="val_acc", save_best_only=True, save_weights_only=False)

hist = model.fit(x_train,y_train,batch_size = 32, epochs = 25, verbose=1,  validation_split=0.2, callbacks=[mcp])
fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_facecolor('w')

ax.grid(b=False)

ax.plot(hist.history['acc'], color='red')

ax.plot(hist.history['val_acc'], color ='green')

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_facecolor('w')

ax.grid(b=False)

ax.plot(hist.history['loss'], color='red')

ax.plot(hist.history['val_loss'], color ='green')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
model.load_weights('model.h5')
from sklearn.metrics import classification_report

pred = model.predict(x_valid)

print(classification_report(np.argmax(y_valid, axis = 1),np.argmax(pred, axis = 1)))

def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.grid(b=False)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.argmax(y_valid, axis = 1),np.argmax(pred, axis = 1))

plot_confusion_matrix(cm = cm,

                      normalize    = False,

                      cmap ='Reds',

                      target_names = ['0','1'],

                      title        = "Confusion Matrix")

# Creating the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.argmax(y_valid, axis = 1),np.argmax(pred, axis = 1))

cm


from keras.preprocessing import image

test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1950_bacteria_4881.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1951_bacteria_4882.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1952_bacteria_4883.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1947_bacteria_4876.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1949_bacteria_4880.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg',target_size=(224,224))



#NORMAL

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1440-0001.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1437-0001.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1436-0001.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1438-0001.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg',target_size=(224,224))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg',target_size=(224,224))



test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=model.predict(test_image)

if result.argmax() == 1:

    prediction = 'PNEMONIA'

    print("The Patient's Lung is affected with ")

    print(prediction)

elif result.argmax() == 0:

    prediction = 'NORMAL'

    print("The Patient's Lung is ")

    print(prediction)
