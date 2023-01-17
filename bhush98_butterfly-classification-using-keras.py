# List of Libraries that we will need



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting

import cv2 # Image reading and preprocessing

import keras # To Build our model

from keras.layers import Conv2D , MaxPooling2D # Getting our Layers for ConvNet

from keras.optimizers import SGD # Our Optimizer, but we will be using adam.

from keras.models import Sequential # We will be using Sequential as our model

from keras.layers import Dropout, Dense , Flatten # Our other layers

# 1 : Dropout :   will switch off some neurons in our model simoultaneously

# 2 : Dense   :   will create a Normal layer of neurons

# 3 : Fatten  :   to Flatten our output from Conv layers 

from keras.utils import to_categorical # to make data categorized like converting data into arrays

from sklearn.model_selection import train_test_split # Splitting the data into training and testing

from matplotlib.image import imread #To read the image

import os



categories = []

# Setting variable filenames to path to iterate better 

filenames = os.listdir("/kaggle/input/butterfly-dataset/leedsbutterfly/images/")

for filename in filenames:

        # Splitting the file.png to get the category 

        # Suppose /kaggle/input/butterfly-dataset/leedsbutterfly/images/001000.png

        category = filename.split(".")[0]

        # This will return 001000

        categories.append(category[0:3])

        # This will append the categories with 001

        

print(categories[0:5])
df = pd.DataFrame({

    "Image" : filenames,

    "Category" : categories

})

df.head()
df.shape
df['Category'].value_counts()
df['Category'].value_counts().plot.bar()
df['Image'].head()
X = []

folder_path = os.listdir("/kaggle/input/butterfly-dataset/leedsbutterfly/images/")

for file in folder_path:

    

    # Reading the Image

    img = cv2.imread("/kaggle/input/butterfly-dataset/leedsbutterfly/images/"+file,cv2.IMREAD_COLOR)

    # Resizing the current Image to a dimension of (128,128,3)

    img = cv2.resize(img,(128,128))

    

    # Converting them to Numpy arrays and appending to our List X

    X.append(np.array(img))

    

# Confirming if Images are converted to our desired dimensions 

print(X[1].shape)

    
df["Category"] = df["Category"].replace({'001': 'Danaus_plexippus', '002': 'Heliconius_charitonius', '003': 'Heliconius_erato', '004': 'Junonia_coenia', '005': 'Lycaena_phlaeas', '006': 'Nymphalis_antiopa', '007': 'Papilio_cresphontes', '008': 'Pieris_rapae', '009': 'Vanessa_atalanta', '010': 'Vanessa_cardui'}) 
y = df['Category'].values

print(y[0:5])
plt.imshow(X[1])
import random as rn

fig,ax=plt.subplots(2,5)

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

fig.set_size_inches(15,15)



for i in range(2):

    for j in range (5):

        l=rn.randint(0,len(y))

        ax[i,j].imshow(X[l][:,:,::-1])

        ax[i,j].set_title(y[l])

        ax[i,j].set_aspect('equal')
df.head()
print(X[0:5])
print(y[0:5])
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

df['Category'] = enc.fit_transform(df['Category'])

print(df.head())
Y = df['Category'].values

print(Y[0:5])

print(Y.ndim)
Y = Y.reshape(len(Y),1)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

Y = ohe.fit_transform(Y)

print(type(Y))
Y.ndim

Y.shape

type(Y)
Y[1].shape
X[1].shape
X = np.array(X)

type(X)
X_train , x_test , Y_train , y_test = train_test_split(X , Y ,test_size = 0.3)
X_train.shape
Y_train.shape
x_test.shape
y_test.shape
model = Sequential()



model.add(Conv2D(32, (5,5), activation = 'relu', input_shape=(128,128,3)))

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(64, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(128, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2, 2)))



model.add(Flatten())



model.add(Dropout(0.4))



model.add(Dense(256, activation='relu'))



model.add(Dense(10, activation='softmax'))
model.summary()
model.layers
model.compile(loss = "categorical_crossentropy" , optimizer = 'adam' , metrics = ['accuracy'])
model.fit(X_train , Y_train , epochs = 30 , batch_size = 12)
loss,accuracy =  model.evaluate(x_test,y_test , batch_size = 32)



print('Test accuracy: {:2.2f}%'.format(accuracy*100))