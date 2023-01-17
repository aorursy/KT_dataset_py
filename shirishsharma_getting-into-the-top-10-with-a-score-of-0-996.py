# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as k

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.applications.vgg16 import VGG16



from random import randint



np.random.seed(0)
train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')
train.head()
print(train.shape)

print(test.shape)
Y = train['label']

X = train.drop('label',axis=1)
total = float(len(X))

plt.figure(figsize=(10,8))

ax = sns.countplot(Y,palette='Set1')

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2. ,height + 40,'{0:.3%}'.format((height/total)),ha="center")

plt.show()
X.isnull().any().describe()
test.isnull().any().describe()
X = X / 255.0

test = test / 255.0
X = X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y = to_categorical(Y,10)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
X_train[0][:,:,0]
g = plt.imshow(X_train[8][:,:,0])
model = keras.models.Sequential([

    keras.layers.Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu',padding='same'),

    keras.layers.BatchNormalization(axis=1), 

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Conv2D(32,(5,5),activation='relu',padding='same'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),

    keras.layers.BatchNormalization(axis=1),  

    keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Flatten(),

    keras.layers.Dense(256,activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(10,activation='softmax')

])
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
epochs = 250

batch_size=64

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,Y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              ,callbacks=[learning_rate_reduction])
plt.figure(figsize=(10,8))

plt.plot(history.history['accuracy'],color='orange')

plt.plot(history.history['val_accuracy'],color='green')

plt.legend(loc='best',shadow=True)

plt.grid()

plt.show()
plt.figure(figsize=(10,8))

plt.plot(history.history['loss'],color='orange')

plt.plot(history.history['val_loss'],color='green')

plt.legend(loc='best',shadow=True)

plt.grid()

plt.show()
def plot_conf_matrix(Y_test,Y_pred):

    conf = confusion_matrix(Y_test,Y_pred)

    recall =(((conf.T)/(conf.sum(axis=1))).T)

    precision =(conf/conf.sum(axis=0))



    print("Confusion Matrix : ")

    class_labels = np.unique(Y)

    plt.figure(figsize=(20,10))

    sns.heatmap(conf,annot=True,fmt=".3f",cmap="GnBu",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("Precision Matrix ; ")

    plt.figure(figsize=(20,10))

    sns.heatmap(precision,annot=True,fmt=".3f",cmap="YlOrBr",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("Recall Matrix ; ")

    plt.figure(figsize=(20,10))

    sns.heatmap(recall,annot=True,fmt=".3f",cmap="Blues",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
Y_pred = model.predict(X_test)

Y_pred_classes = np.argmax(Y_pred,axis=1)

Y_true = np.argmax(Y_test,axis=1)

plot_conf_matrix(Y_true,Y_pred_classes)
errors = (Y_pred_classes - Y_true != 0)

fig,ax = plt.subplots(5,5,figsize=(20,20))

for i in range(5):

    for j in range(5):

        l = randint(0,9)

        ax[i,j].imshow((X_test[errors][l]).reshape(28,28))

        ax[i,j].set_title("Predicted label : {} , True label : {}".format(Y_pred_classes[errors][l],Y_true[errors][l]))

plt.tight_layout()
predictions = model.predict(test)

predictions = np.argmax(predictions,axis = 1)

predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)

submission.to_csv("submissions_mnist.csv",index=False)

print("Your file is saved.")