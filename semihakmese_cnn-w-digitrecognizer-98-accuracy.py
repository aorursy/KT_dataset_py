# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 



import warnings 

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing ,Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading Datas

train = pd.read_csv("/kaggle/input/digitrecognizer/train.csv")

test = pd.read_csv("/kaggle/input/digitrecognizer/test.csv")
print(train.shape) #42000 image size, 784pix (24x24) and 1 label 

train.head()
print(test.shape) #28000 image size 

test.head()
Y_train=train["label"]

X_train =train.drop(labels=["label"],axis=1)
#Visualising number of Digits Class 

print(Y_train.value_counts())

plt.figure(figsize=(15,8))

sns.countplot(Y_train,palette = "RdYlBu")

plt.title("Number of Digits' Label Pixels")
#plotting some of the samples  

img = X_train.iloc[0].to_numpy()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[0,0])

plt.axis("off")

plt.show()
img = X_train.iloc[3].to_numpy()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[3,0])

plt.axis("off")

plt.show()
X_train = X_train / 255.0

test = test / 2500.0

print("X_train shape :",X_train.shape)

print("Test Shape :",test.shape)
#Reshape 

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("X_train shape: ",X_train.shape)

print("Test shape: ",test.shape)
#Label Encoding - Keras provied encoding like we showed in upper description

from keras.utils.np_utils import to_categorical #Convert to one hot encoding 

Y_train = to_categorical(Y_train,num_classes = 10)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.1, random_state = 2)

print("X_train shape",X_train.shape)

print("X_val shape",X_val.shape)

print("Y_train shape",Y_train.shape)

print("Y_val shape",Y_val.shape)
from sklearn.metrics import confusion_matrix 

import itertools 



from keras.utils.np_utils import to_categorical #Converting to one hot encoding 

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()

#

model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1))) #8 filtre 5x5 boyutlu 

                #ilk shape verilmeli 

model.add(MaxPool2D(pool_size=(2,2))) #Pooling size küçülttü ve yoğunlaştırdı

                                    

model.add(Dropout(0.25)) #nodeların yüzde 25 i deactive (ezberi engeller)

#

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

# fully connected

model.add(Flatten())

model.add(Dense(256, activation = "relu")) #Hidden layer 1

model.add(Dropout(0.5))

model.add(Dense(128, activation = "relu")) #Hidden layer 2

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax")) #Softmax Sigmoid in genelleşmiş halidir. 

                                            #Sigmoid binary classification(2 label) yaparken softmax multi class 
optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2= 0.999)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

#ikiden fazla class varsa categorical crossentropy kullanılır 

#Binary de ise 2 class vardı 
epochs = 15

batch_size = 250  #Epochs arttıkca ve Batch size küçüldükce işlem süresi artar ancak accuracy de artabilir

#Çünkü batch size küçüldükçe her aşamada daha az veri tarar ve bu nedenle toplam iterasyon artar 
#data augmentation 

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.2,  # randomly rotate images in the range 5 degrees

        zoom_range = 0.2, # Randomly zoom image 5%

        width_shift_range=0.2,  # randomly shift images horizontally 5%

        height_shift_range=0.2,  # randomly shift images vertically 5%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)    
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size),

                             epochs = epochs, validation_data = (X_val,Y_val),

                             steps_per_epoch = X_train.shape[0] // batch_size)
# BU ŞEKİLDE VE BATCH SIZE 150 epochs 20 iken değerler Yaklaşık Yüzde 95 Doğrulukta oluyor 

# BU SEKİLDE BATCH SIZE 250 ve Epochs 15 + 1 hidden layer (Yani 2 Hidden layer) daha eklenince Yüzde 94 doğrulukta oluyor 

# BU SEKİLDE BATCH SIZE VE EPOCHS BİR ONCEKİ İLE AYNI İKEN DATA AUGMENTATION KISMINDAKİ O.5 leri küçültelim Sonuç Yüzde 97.6 ya ulaştık Yani Data Augmentation da işlemi sıkılaştırdıkça modelimizin accuracy artıyor 





# model = Sequential()

# #

# model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

#                  activation ='relu', input_shape = (28,28,1))) #8 filtre 5x5 boyutlu 

#                 #ilk shape verilmeli 

# model.add(MaxPool2D(pool_size=(2,2))) #Pooling size küçülttü ve yoğunlaştırdı

#                                     

# model.add(Dropout(0.25)) #nodeların yüzde 25 i deactive (ezberi engeller)

# #

# model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

#                  activation ='relu'))

# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# model.add(Dropout(0.25))

# # fully connected

# model.add(Flatten())

# model.add(Dense(256, activation = "relu")) #Hidden layer 1

# model.add(Dropout(0.5))

# model.add(Dense(10, activation = "softmax")) #Softmax Sigmoid in genelleşmiş halidir. 

#                                             #Sigmoid binary classification(2 label) yaparken softmax multi class 
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# confusion matrix

import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()