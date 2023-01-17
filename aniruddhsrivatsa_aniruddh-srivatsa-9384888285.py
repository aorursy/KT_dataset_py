
CATEGORIES =["Ac","As","Cb","Cc","Ci","Cs","Ct","Cu","Ns","Sc","St"]

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_size=50

training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join("../input/cloud-classification/data/train/",category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                img_array=cv2.resize(img_array,(img_size,img_size))
                
                training_data.append([img_array,class_num])
                
            except Exception as e:
                pass
create_training_data()

X=[]
Y=[]
for dat,lab in training_data:
  X.append(dat)
  Y.append(lab)
X=np.array(X)
Y=np.array(Y) 
X=X/25
import pickle
pickle.dump(X,open("X.pkl","wb"))
pickle.dump(Y,open("Y.pkl","wb"))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y)
print(X_train.shape)
print(y_train.shape)
X_train=X_train.reshape((-1,50,50,1))
X_test=X_test.reshape((-1,50,50,1))
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,GlobalAveragePooling2D,Convolution2D,ZeroPadding2D,Convolution2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
callback=EarlyStopping(patience=4,monitor="val_loss")
def my_VG():
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(X_train.shape[1:])))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))

   
    model.add(Conv2D(256, (3, 3), activation='relu'))
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    

    
    model.add(Conv2D(256, (3, 3), activation='relu'))
   
    model.add(Conv2D(256, (3, 3), activation='relu'))
    
    
    

    
    
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    

    
   
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
   
    model.add(Conv2D(128, (3, 3), activation='relu'))
   
    
    
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(11, activation='sigmoid'))

    

    return model

from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Dropout
model = my_VG()

model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy',metrics=["accuracy"])
#CONV Layer1
model=Sequential()
model.add(Conv2D(256,(3,3),activation = 'relu'))
model.add(MaxPool2D(2,2))
#CONV Layer2
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPool2D(2,2))
#CONV Layer3
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPool2D(2,2))
#CONV Layer4
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPool2D(2,2))
#FLATTERN
model.add(Flatten())
#DENSE LAYER
model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
#O/P LAYER
model.add(Dense(11, activation = 'sigmoid'))
#COMPILE
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history11=model.fit(X_train,y_train,validation_split=0.1,epochs=50,callbacks=[callback])
model.evaluate(X_test,y_test)
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history11.history[string])
  plt.plot(history11.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
plot_graphs(history11, 'accuracy')
plot_graphs(history11, 'loss')
    
CATEGORIES =["Ac","As","Cb","Cc","Ci","Cs","Ct","Cu","Ns","Sc","St"]

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_size=50

test_data=[]
def create_test_data():
    
        path=os.path.join("../input/cloud-classification/data/test/")
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                img_array=cv2.resize(img_array,(img_size,img_size))
                
                test_data.append([img_array])
                
            except Exception as e:
                pass
create_test_data()
len(test_data)
test_data=np.array(test_data)
test_data=test_data/255
test_data=test_data.reshape(-1,50,50,1)
y_pred=model.predict(test_data)
asn=[]
for i in range(len(y_pred)):
 c=np.argmax(y_pred[i])
 asn.append(CATEGORIES[c])

import pandas as pd
dff=pd.read_csv("../input/cloud-classification/data/submission.csv")
dff.head()
dff.drop("Class",inplace=True,axis=1)
dff["Class"]=asn
dff.head()
dff.to_csv("submission3.csv",index=False)