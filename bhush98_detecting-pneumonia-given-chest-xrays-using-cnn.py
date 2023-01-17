# Importing all the libraris we need



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import cv2

import matplotlib.pyplot as plt

from matplotlib.image import imread

from keras.models import Sequential

from keras.layers import Conv2D , MaxPooling2D , Dropout

from keras.layers import Dense , Flatten 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.image import ImageDataGenerator , load_img

from keras.optimizers import SGD

import os
for dirname, _, filenames in os.walk("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray"):

    print(dirname)



# Dividing the folders, as per training , testing , Normal , Pneumonia 

chest_xray_train_normal = os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/")

chest_xray_train_pneumonia = os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/")

chest_xray_test_normal = os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL")

chest_xray_test_pneumonia = os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA")



# To store the file names and category

filenames = []

Category = []



# Appending all the normal images to filenames and category to Category

for file in chest_xray_train_normal:

    filenames.append(file)

    Category.append("NORMAL")

    

# Appending all the pneumonia images to filenames and category to Category

for file in chest_xray_train_pneumonia:

    filenames.append(file)

    Category.append("PNEUMONIA")

    

# To find the total number of observations we have

print(len(filenames))

print(len(Category))
# Creating a Dataframe although we are not going to use one, just to show how you can create a dataframe 



df_train = pd.DataFrame({

    "Files" : filenames,

    "Category" : Category

})



df_train.head(10)
df_train.tail(10)
# Same thing as training, but this is for testing



test_files = []

test_category = []

for file in chest_xray_test_normal:

    test_files.append(file)

    test_category.append("NORMAL")



for file in chest_xray_test_pneumonia:

    test_files.append(file)

    test_category.append("PNEUMONIA")

    

print(len(test_files))

print(len(test_category))
df_test = pd.DataFrame({

    "Files" : test_files,

    "Category" : test_category

})



df_test.head(10)
df_test.tail(10)
# It's better to know the dimensions of the image before preprocessing



#to read the image

img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-0569-0001.jpeg")



#this will return dimensions of the image and store it in dimensions variable 

dimensions = img.shape



# Printing the dimensions 

print("Dimensions : ",dimensions)

print("Height : ",dimensions[0])

print("Width : ",dimensions[1])

print("Channels : ",dimensions[2])
# Plotting image of a Chest X-ray that is Normal



image_title = "NORMAL"

img = load_img("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-0569-0001.jpeg")



plt.imshow(img)

plt.title(image_title)

plt.show()
# Plotting image of a Chest X-ray that is affected by Pneumonia 



image_title = "PNEUMONIA"

img = load_img("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/person675_bacteria_2569.jpeg")



plt.imshow(img)

plt.title(image_title)

plt.show()
# Converting the images from original dimensions to (128,128,3)



X_train = []

Y_train = []

x_test = []

y_test = []



for file in chest_xray_train_normal:

    try:

        img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/"+file,cv2.IMREAD_COLOR)

        img = cv2.resize(img,(128,128))

    

        X_train.append(np.array(img))

        Y_train.append("NORMAL")



    except:

        pass

        

for file in chest_xray_train_pneumonia:

    try :

        img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/"+file,cv2.IMREAD_COLOR)

        img = cv2.resize(img,(128,128))

    

        X_train.append(np.array(img))

        Y_train.append("PNUEMONIA")

        

    except: 

        pass 



for file in chest_xray_test_normal:

    

    try:

        #print(file)

        img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/"+file,cv2.IMREAD_COLOR)

        img = cv2.resize(img,(128,128))

    

        x_test.append(np.array(img))

        y_test.append("NORMAL")

    

    except:

        pass

    

for file in chest_xray_test_pneumonia:

    

    try:

        img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/"+file,cv2.IMREAD_COLOR)

        img = cv2.resize(img,(128,128))

    

        x_test.append(np.array(img))

        y_test.append("PNUEMONIA")    

    

    except:

        pass



print("Total size of X_train is: ",len(X_train))

print("Total size of Y_train is: ",len(Y_train))

print("Total size of x_test is: ",len(x_test))

print("Total size of y_test is: ",len(y_test))
dimensions = X_train[1].shape

print("The shape of Image is : ",dimensions)

print("Height of Image is : ",dimensions[0])

print("Width of Image is : ",dimensions[1])

print("Number of channels : ",dimensions[2])
# This variables were lists we need to convert them to numpy arrays



X_train = np.array(X_train)

x_test = np.array(x_test)

Y_train = np.array(Y_train)

y_test = np.array(y_test)
Y_train.shape
# Plotting some random images



import random

fig,ax = plt.subplots(2,5)

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

fig.set_size_inches(15,15)



for i in range(2):

    for j in range(5):

        l = random.randint(0,len(Y_train))

        ax[i,j].imshow(X_train[l])

        ax[i,j].set_title(str(Y_train[l]))

        ax[i,j].set_aspect('equal')
print(Y_train.shape)

print(y_test.shape)
# Encoding the categories using OneHotEncoder

# As we have two classes here "NORMAL" , "PNEUMONIA", so the one hot encoded values will be

# NORMAL : [1,0]

# PNEUMONIA : [0,1]



enc = LabelEncoder()

Y_train = enc.fit_transform(Y_train)

y_test = enc.fit_transform(y_test)
Y_train.shape
y_test.shape
Y_train[0:5]
x_test.shape
sgd = SGD(lr = 0.1 , decay = 1e-2 , momentum = 0.9 )
model = Sequential()



model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(128,128,3)))

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(64, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(128, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(256 , (3,3) , activation = 'relu'))



model.add(Flatten())



model.add(Dense(256, activation='relu'))



model.add(Dropout(0.5))



model.add(Dense(1, activation='sigmoid'))
# This will give us a summary of our model



model.summary()
model.compile( loss = "binary_crossentropy" , optimizer  = 'adam' , metrics = ['accuracy'])
# Training our model 



model.fit(X_train,Y_train,epochs = 30, batch_size = 32)
# Evaluating our model on testing data



loss , accuracy = model.evaluate(x_test , y_test , batch_size = 32)



print('Test accuracy: {:2.2f}%'.format(accuracy*100))