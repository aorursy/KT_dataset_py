#importing liberaries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
#data path
trainpath = '../input/intel-image-classification/seg_train/'
testpath = '../input/intel-image-classification/seg_test/'
predpath = '../input/intel-image-classification/seg_pred/'
#training data informations 
Folder_name=[]
folder_item_numbers = []
for folder in  os.listdir(trainpath + 'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    Folder_name.append(folder)
    folder_item_numbers.append(len(files))
foldernames=pd.DataFrame({'Folder_name':Folder_name})
itemnumbers=pd.DataFrame({'Traning Image Numbers':folder_item_numbers})
informations=pd.concat([foldernames,itemnumbers],axis=1)
print(informations)

#test data informations 
Folder_name=[]
folder_item_numbers = []
for folder in  os.listdir(testpath + 'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))
    Folder_name.append(folder)
    folder_item_numbers.append(len(files))
foldernames=pd.DataFrame({'Folder_name':Folder_name})
itemnumbers=pd.DataFrame({' Test Image Numbers':folder_item_numbers})
informations=pd.concat([foldernames,itemnumbers],axis=1)
print(informations)
#prediction data informations 
Folder_name=[]
folder_item_numbers = []
for folder in  os.listdir(predpath) : 
    files = gb.glob(pathname= str( predpath + folder + '/*.jpg'))
    Folder_name.append(folder)
    folder_item_numbers.append(len(files))
foldernames=pd.DataFrame({'Folder_name':Folder_name})
itemnumbers=pd.DataFrame({' pred Image Numbers':folder_item_numbers})
informations=pd.concat([foldernames,itemnumbers],axis=1)
print(informations)
#checking image size for traning data
Image_size = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    for image in files: 
        read_image = plt.imread(image)
        Image_size.append(read_image.shape)
pd.Series(Image_size).value_counts()
#checking image size for test data
Image_size = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))
    for image in files: 
        read_image = plt.imread(image)
        Image_size.append(read_image.shape)
pd.Series(Image_size).value_counts()
#checking image size for pred data
Image_size = []
for folder in  os.listdir(predpath) : 
    files = gb.glob(pathname= str( predpath + folder + '/*.jpg'))
    for image in files: 
        read_image = plt.imread(image)
        Image_size.append(read_image.shape)
pd.Series(Image_size).value_counts()
#resize each image in all folders
#identifing new size as 100 
#converting images to an array as X_train and and making a labeling array for it as y_train
new_size=100    
X_train = []
y_train = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    for file in files: 
        image_class = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
        orignal_image = cv2.imread(file)
        resized_image = cv2.resize(orignal_image , (new_size,new_size))
        X_train.append(list(resized_image))
        y_train.append(image_class[folder])

#check items in X_train
print("items in X_train is:       ",len(X_train) , " items")
#showing training images with labels
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    classes = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
    def get_img_class(n):
        for x , y in classes.items():
            if n == y :
                return x
    plt.title(get_img_class(y_train[i]))
#resize each image in all folders for Test Data
#identifing new size as 100 
#converting images to an array as X_test and and making a labeling array for it as y_test
new_size=100    
X_test = []
y_test = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))
    for file in files: 
        image_class = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
        orignal_image = cv2.imread(file)
        resized_image = cv2.resize(orignal_image , (new_size,new_size))
        X_test.append(list(resized_image))
        y_test.append(image_class[folder])
#check items in X_test
print("items in X_test is:       ",len(X_test) , " items")
#showing test images with labels
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])   
    plt.axis('off')
    classes = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
    def get_img_class(n):
        for x , y in classes.items():
            if n == y :
                return x
    plt.title(get_img_class(y_test[i]))
#resize each image in all folders for prediction Data
#identifing new size as 100 
#converting images to an array as X_pred
new_size=100    
X_pred = []
for folder in  os.listdir(predpath) : 
    files = gb.glob(pathname= str( predpath + folder + '/*.jpg'))
    for file in files: 
        image_class = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
        orignal_image = cv2.imread(file)
        resized_image = cv2.resize(orignal_image , (new_size,new_size))
        X_pred.append(list(resized_image))
#check items in X_pred
print("items in X_pred is:       ",len(X_pred) , " items")
#showing some prediction images
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])   
    plt.axis('off')
#converting all data to array
X_train = np.array(X_train)
X_test = np.array(X_test)
X_Pred = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)
print("X_train shape  : ",X_train.shape)
print("X_test shape  :" ,X_test.shape)
print("X_Pred shape :" , X_Pred.shape)
print("y_train shape :" ,y_train.shape)
print("y_test shape :", y_test.shape)

Classification_Model_Keras = keras.models.Sequential([
        keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu',input_shape=(new_size,new_size,3)),
        keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu'),
        keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten() ,    
        keras.layers.Dense(128,activation='relu') ,    
        keras.layers.Dense(64,activation='relu') ,    
        keras.layers.Dense(32,activation='relu') ,        
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(6,activation='softmax') ,    
        ])
Classification_Model_Keras.compile(optimizer ='adam',
                                   loss='sparse_categorical_crossentropy',
                                   metrics=['accuracy'])
print('Model Summary: ')
print(Classification_Model_Keras.summary())
epochs = 40
KerasModel = Classification_Model_Keras.fit(X_train, y_train, epochs=epochs,batch_size=64,verbose=1)

val_Loss, val_Acc = Classification_Model_Keras.evaluate(X_test, y_test)

print('Test Loss:', val_Loss)
print('Test Accuracy :', val_Acc)
y_test_pred = Classification_Model_Keras.predict(X_test)

print('y_test_pred Shape :',y_test_pred.shape)
y_pred = Classification_Model_Keras.predict(X_Pred)

print('Prediction Shape for y_result : ',y_pred.shape)
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')
    classes = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
    def get_img_class(n):
        for x , y in classes.items():
            if n == y :
                return x
    plt.title(get_img_class(np.argmax(y_pred[i])))
