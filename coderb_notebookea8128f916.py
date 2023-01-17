# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # for image proccesing

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
X_train = []
y_train = []
X_test = []
y_test = []
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    i = 0
    for filename in filenames:
        im = cv2.imread(os.path.join(dirname, filename))
        im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        rect = rects[0]
        im_crop =im_th[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        im_resize = cv2.resize(im_crop,(28,28))
        im_resize = np.array(im_resize)
        im_resize=im_resize.reshape(28,28)
        ya = [0,0,0,0,0,0,0,0,0,0,0,0]
        try:
            ya[int(dirname[-2:])] = 1
        except:
            ya[int(dirname[-1:])] = 1
        if i%19 == 0 or i%18==0 or i%17==0 or i%16==0 or i%15 == 0:
            X_test.append(im_resize)
            y_test.append(ya)
        else:
            X_train.append(im_resize)
            y_train.append(ya)
        i+=1
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_test.shape,X_train.shape)
X_train = X_train.reshape(168,28,28,1)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import numpy as np 
import pickle 
np.random.seed(1212) 
import keras 
from keras.models import Model 
from keras.layers import *
from keras import optimizers 
from keras.layers import Input, Dense 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.layers import Flatten 
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D 
from keras.utils import np_utils 
from keras import backend as K  
from keras.utils.np_utils import to_categorical 
from keras.models import model_from_json 
model = Sequential() 
model.add(Conv2D(30, (5, 5), input_shape =(28,28,1), activation ='relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
model.add(Conv2D(15, (3, 3), activation ='relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
model.add(Dropout(0.2)) 
model.add(Flatten()) 
model.add(Dense(128, activation ='relu')) 
model.add(Dense(50, activation ='relu')) 
model.add(Dense(12, activation ='softmax')) 
# Compile model 
model.compile(loss ='categorical_crossentropy', 
			optimizer ='adam', metrics =['accuracy']) 
model.fit(X_train, y_train, epochs = 500, batch_size = 200,  
          shuffle = True, verbose = 1) 
X_test = X_test.reshape(72,28,28,1)
model.evaluate(X_test,y_test)
