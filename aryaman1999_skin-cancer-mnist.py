# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten,MaxPooling2D,BatchNormalization,Dropout
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.preprocessing import image
from keras.models import Model
from tensorflow.keras.metrics import top_k_categorical_accuracy
import cv2
import glob
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/hmnist_28_28_RGB.csv')
data.info(null_counts=True)
def UnderSampling():
    new_data_set = data[data['label']==3]

    
    for i in range(7):
        ind = data[data['label']==i]
        if(i not in [1,0,3,5]):
            
            sample_ind = ind.sample(n=800)
        else:
            sample_ind = ind
        new_data_set = new_data_set.append(sample_ind)
        
    return new_data_set

        
data_train = UnderSampling()

data_train.info(null_counts=True)
#metadata = pd.read_csv('../input/HAM10000_metadata.csv')
Y = data['label']
data.drop('label',axis=1,inplace=True)
data.head()
Y = to_categorical(Y,num_classes=7)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data,Y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[5,10,20],'min_samples_split':[2,50,100,150]}
GS = GridSearchCV(RF,parameters,cv=5)
GS.fit(X_train,Y_train)


pred = GS.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,Y_test)

#print(classification_report(pred,Y_test))
model = Sequential()
model.add(Dense(2352,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(250,activation = 'relu'))

model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())

model.add(Dense(50,activation='relu'))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(np.array(X_train),np.array(Y_train),epochs=50,batch_size=10)
model.save("skin_cancer.hd5")