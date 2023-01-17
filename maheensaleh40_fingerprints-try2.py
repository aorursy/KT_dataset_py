# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/working/'):
    for filename in filenames:
        name = os.path.join(dirname, filename)
        print(os.path.join(dirname, filename))
#         ind = name.index("__")
#         print(name[ind+2])

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.models import load_model
# from imutils import paths
import numpy as np
import random
import cv2
import os
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten,Activation,  Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# from imutils import paths
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

training_data =  []


# path = "C:/Users/uni tech/Desktop/spyderr/datasets/SOCOFing/Real"
path = "/kaggle/input/socofing/socofing/SOCOFing/Real"


for img in os.listdir(path):
    list_of_strings=[]
    img_path = os.path.join(path,img)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (80, 80))

    
    new_name=os.path.split(img_path)[-1]
    new_name2 = new_name[:-4]
   
    for x in new_name2:
        list_of_strings.append(x)
 
    
    if "M" in list_of_strings:
        training_data.append([new_array, 0])
        
    elif "F" in list_of_strings:
        training_data.append([new_array, 1])
   
        

        
random.shuffle(training_data)


for sample in training_data:
    print(sample[1])
    

X=[]
y=[]

for features, labels in training_data:
    X.append(features)
    y.append(labels)
    

X = np.array(X).reshape(-1, 80, 80, 1)
X = X / 255
# Defining callbacks function    
early_stoppings = EarlyStopping(monitor='val_loss',
                                patience = 3,
                                verbose = 1,
                                restore_best_weights = True)   


# Defining the model and adding layers to it
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(500))
model.add(Activation("relu"))


model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(1))
model.add(Activation("sigmoid"))


print(model.summary())
# model compilation
# adam = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer = "adam", loss="binary_crossentropy",  metrics=['accuracy'])
## convert y to np array type
print(type(y))
y = np.array(y)
print(type(y))

# # Model training
# model.fit(X, y ,batch_size=100 ,epochs = 10, validation_split=0.1 , callbacks= [early_stoppings])
model.fit(X, y ,batch_size=100 ,epochs = 20, validation_split=0.1 )

# from keras.models import load_model 
model.save('fingerprint_recog.h5')
# model = load_model('fingerprint_recog.h5')
model = load_model('/kaggle/working/fingerprint_recog.h5')


# path = "C:/Users/uni tech/Desktop/spyderr/datasets/SOCOFing/Altered/Altered-Easy"
path = "/kaggle/input/socofing/socofing/SOCOFing/Altered/Altered-Easy"
n=0
testing_data =  []


for img in os.listdir(path):
    if n<500:
        list_of_strings=[]
        img_path = os.path.join(path,img)
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (80, 80))
        
        new_name=os.path.split(img_path)[-1]
        new_name2 = new_name[:-4]
   
        for x in new_name2:
            list_of_strings.append(x)
     
        
        if "M" in list_of_strings:
            testing_data.append([new_array, 0])
            
        elif "F" in list_of_strings:
            testing_data.append([new_array, 1])
            # X_test.append(new_array)
        n+=1
        
        
    else:
        break

random.shuffle(testing_data)


X_test=[]
y_test=[]

for features, labels in testing_data:
    X_test.append(features)
    y_test.append(labels)
    

X_test = np.array(X_test).reshape(-1, 80, 80, 1)
X_test = X_test / 255




result = model.predict(X_test)
classes = np.round(result)

# classes = np.argmax(result, axis = 1)
# print(classes)

accuracy = r2_score(y_test, result)
print(accuracy)
