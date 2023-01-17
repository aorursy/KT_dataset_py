

import seaborn as sns
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

paths_normal = []
paths_pneumonia = []

import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/"):
    for filename in filenames:
        paths_normal.append(os.path.join(dirname, filename))
    
import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"):
    for filename in filenames:
        paths_pneumonia.append(os.path.join(dirname, filename))
    
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
label_normal = np.zeros((500,1))
label_pneumonia = np.ones((500,1))
label = list(np.concatenate((label_normal,label_pneumonia),axis = 0));
paths = paths_normal[0:500] + paths_pneumonia[0:500]

d = {'paths': paths, 'label': label
    }
df = pd.DataFrame(data=d)


X = np.zeros((1,100*100),np.uint8)
y = np.zeros((1,1),np.uint8)
for count,ele in enumerate (df.iloc[:,0],0): 
    y_temp = df.iloc[count,1]
    y = np.vstack((y,y_temp))
    X_temp = cv.imread(ele,cv.IMREAD_GRAYSCALE) 
    X_temp = cv.resize(X_temp,(100,100)).reshape(1,100*100)
    X = np.vstack((X,X_temp))
    print("progression : %{}".format((count/10)))
    if count/10 >= 99.9:
        print("Done")
X = X[1:,:]
y = y[1:,:]
        


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

plt.figure(figsize=(15,15))
for count,i in enumerate(range(0,6),231):
    
    plt.subplot(count)
    if y_train[i]==1:
        plt.title("Pneumonia")
        plt.imshow(X_train[i,:].reshape(100,100),'gray')
        
    elif y_train[i]==0:
        plt.title("Normal")
        plt.imshow(X_train[i,:].reshape(100,100),'gray')
plt.show()        
isnan_train = np.isnan(X_train).all()
isnan_test = np.isnan(X_val).all()
print(isnan_train,isnan_test)


X_train,X_val = X_train[:,:]/255, X_val[:,:]/255


#Model
model = tf.keras.Sequential()
model.add(Dense(units = 784/2, activation = 'relu', input_dim=X_train.shape[1]))
model.add(Dense(units = 784/4, activation = 'relu'))
model.add(Dense(units = 784/8, activation = 'relu'))
model.add(Dense(units = 784/16, activation = 'relu'))
model.add(Dense(units = 784/32, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(loss="binary_crossentropy",optimizer="sgd", metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=20, epochs=90)
#Making Predictions on Test data
predicted = model.predict(X_val)
y_head_ann = [0 if i<0.5 else 1 for i in predicted]

print(accuracy_score(y_val, y_head_ann))
cm_ann = confusion_matrix(y_val,y_head_ann)
sns.heatmap(cm_ann, annot=True) ;
 

#Initialising the CNN
cnn = Sequential()
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[100,100,1]))  # 1 is our canal number it is just 1 because we use grayscale data
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu'))

#Pooling
cnn.add(layers.MaxPool2D(pool_size=2,strides=2)) #I preffered Max Pooling for this model
cnn.add(Dropout(0.2))

#Second Layer
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(Dropout(0.2))



#Flattening and bulding ANN

cnn.add(Flatten())
cnn.add(Dense(64, activation = "relu"))
cnn.add(Dense(32, activation = "relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation = "sigmoid")) 

# Now we need to choose loss function, optimizer and compile the model
cnn.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

X_train = X_train.reshape(-1,100,100,1)
X_val = X_val.reshape(-1,100,100,1)


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
        horizontal_flip=False,  
        vertical_flip=False)  


datagen.fit(X_train)
cnn.fit_generator(datagen.flow(X_train,y_train, batch_size=20),epochs = 90, validation_data = (X_val,y_val),verbose = 1,steps_per_epoch=len(X_train) // 20)
predicted = cnn.predict(X_val)

y_head_cnn = [0 if i<0.5 else 1 for i in predicted]

print(accuracy_score(y_val, y_head_cnn))
cm_cnn = confusion_matrix(y_val,y_head_cnn)
sns.heatmap(cm_cnn, annot=True) ;
plt.figure(figsize=(5, 5))
plt.subplot(221)
plt.title("ANN Confusion Matrix")
sns.heatmap(cm_ann, annot=True) ;

plt.subplot(222)
plt.title("CNN Confusion Matrix")
sns.heatmap(cm_cnn, annot=True) ;
plt.show()
paths_normal_test = []
paths_pneumonia_test = []

import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/"):
    for filename in filenames:
        paths_normal_test.append(os.path.join(dirname, filename))
    
import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"):
    for filename in filenames:
        paths_pneumonia_test.append(os.path.join(dirname, filename))

label_normal_test = np.zeros((500,1))
label_pneumonia_test = np.ones((500,1))
label_test = list(np.concatenate((label_normal_test,label_pneumonia_test),axis = 0));
paths_test = paths_normal_test[0:500] + paths_pneumonia_test[0:500]

d = {'paths': paths_test, 'label': label_test
    }
df_test = pd.DataFrame(data=d)

X = np.zeros((1,100*100),np.uint8)
y = np.zeros((1,1),np.uint8)
for count,ele in enumerate (df.iloc[:,0],0): 
    y_temp = df.iloc[count,1]
    y = np.vstack((y,y_temp))
    X_temp = cv.cvtColor(cv.imread(ele),cv.COLOR_BGR2GRAY)  
    X_temp = cv.resize(X_temp,(100,100)).reshape(1,100*100)
    X = np.vstack((X,X_temp))
    print("progression : %{}".format((count/10)))
    if count/10 >= 99.9:
        print("Done")
X_test = X[1:,:]
y_test = y[1:,:]
        
X_test = X_test[:,:]/255
X_test = X_test.reshape(-1,100,100,1)
predicted = cnn.predict(X_test)
y_head = [0 if i<0.5 else 1 for i in predicted]

print(accuracy_score(y_test, y_head))
cm = confusion_matrix(y_test,y_head)
sns.heatmap(cm, annot=True) ;