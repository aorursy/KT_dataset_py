#import the necessary libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from skimage.io import imshow



from keras.layers import Dense

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.models import Sequential
#Reading dataset

X = pd.read_csv("../input/deepsat-sat4/X_test_sat4.csv") #values are in DataFrame format

Y = pd.read_csv("../input/deepsat-sat4/y_test_sat4.csv") #values are in DataFrame format

X = np.array(X) # converting Dataframe to numpy array

Y = np.array(Y) # converting Dataframe to numpy array
#Shape of data used

print("Train data shape: ",X.shape)
#reshaping (99999, 3136) to (99999, 28, 28, 4)

X = X.reshape([99999,28,28,4]).astype(float)

print("Reshaped data format: ",X.shape)
#splitting data into train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0) 
#format of train and test data

print("X train data shape: ",x_train.shape)

print("Y train data shape: ",y_train.shape)

print("X test data shape: ",x_test.shape)

print("Y test data shape: ",y_test.shape)
#normalizing train and test data

x_train = x_train/255

x_test = x_test/255
#Images in the data with its label(reduced image)

img_no = 1276 #type a random number in inclusive range 0 to 79999

imshow(np.squeeze(x_train[img_no,:,:,0:3]).astype(float)) #taking only RGB format

plt.show()

print ('Ground Truth: ',end='')

if y_train[img_no, 0] == 1:

    print ('Barren Land')

elif y_train[img_no, 1] == 1:

    print ('Forest Land')

elif y_train[img_no, 2] == 1:

    print ('Grassland')

else:

    print ('Other')
#defining layers

num_classes = 4

from keras.layers.advanced_activations import LeakyReLU

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,4),padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D((2, 2),padding='same'))

model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

model.add(LeakyReLU(alpha=0.1))                  

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Flatten())

model.add(Dense(128, activation='linear'))

model.add(LeakyReLU(alpha=0.1))                  

model.add(Dense(num_classes, input_shape=(3136,), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#CNN Model summary

model.summary()
#fitting the data into the model

model.fit(x_train,y_train,batch_size=64, epochs=20, verbose=1, validation_split=0.20)
#predicting model performance

preds = model.predict(x_test, verbose=1)
img_no = 587#Type a number between 0 and 20000 inclusive

imshow(np.squeeze(x_test[img_no,:,:,0:3]).astype(float)) #Only seeing the RGB channels

plt.show()

#Predicted classification

print ('Predicted Label: ',end='')

if preds[img_no, 0]*100  >= 80:

    print ('Barren Land')

elif preds[img_no, 1]*100 >= 80:

    print ('Forest Land')

elif preds[img_no, 2]*100 >= 80:

    print ('Grassland')

else:

    print ('Other')



#Acutal classification

print ('Actual label: ',end='')

if y_test[img_no, 0] == 1:

    print ('Barren Land')

elif y_test[img_no, 1] == 1:

    print ('Forest Land')

elif y_test[img_no, 2] == 1:

    print ('Grassland')

else:

    print ('Other')
#model performance evaluation

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy score: ",accuracy_score(y_test, np.round_(preds)))

print("Classification report:")

print(classification_report(y_test, np.round_(preds)))
print("Accuracy of CNN model is: ", accuracy_score(y_test,np.round_(preds))*100)