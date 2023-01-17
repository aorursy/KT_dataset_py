# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import cv2
import os
import glob as gb
import tensorflow as tf
import seaborn as sns
import keras
TrainPath = "../input/intel-image-classification/seg_train/seg_train"
TestPath  = "../input/intel-image-classification/seg_test/seg_test"
PredPath = "../input/intel-image-classification/seg_pred/seg_pred"
for folder in os.listdir(TrainPath):
    files = gb.glob(pathname = str(TrainPath + "/"+ folder + '/*.jpg'))
    print("Found in {}    {} fiels".format(folder , len(files)))
PredFiles = gb.glob(pathname = str(PredPath + "/*jpg"))
print("ALL FIELS IN PRED FOLDER ARE " , len(files))
code = {"street": 0 , "forest":1 , "mountain":2 , "buildings":3 , "glacier":4 , "sea":5}
def GetCode(n):
    for x , y in code.items():
        if n==y:
            return  x
import matplotlib.pyplot as plt
import pandas as pd
size = []
for file in files:
    img = plt.imread(file)
    size.append(img.shape)
pd.Series(size).value_counts()
s = 100
X_train = list()
y_train = list()

for folder in os.listdir(TrainPath):
    Files = gb.glob(pathname = str(TrainPath + "/" + folder + "/*jpg"))
    for file in Files:
        img = cv2.imread(file)
        image_array = cv2.resize(img , (s , s))
        X_train.append(list(image_array))
        y_train.append(code[folder])
np.array(X_train).shape
plt.figure(figsize = (20 , 20))
for n , i in enumerate(list(np.random.randint(0 , len(X_train) , 42) )):
    plt.subplot(6 , 7 , n+1)
    plt.imshow(X_train[i])
    plt.axis("off")
    plt.title(GetCode(y_train[i]))
X_test = list()
y_test = list()

for folder in os.listdir(TestPath):
    Files = gb.glob(pathname = str(TestPath + "/" + folder + "/*jpg"))
    for file in Files:
        img = cv2.imread(file)
        image_array = cv2.resize(img , (s , s))
        X_test.append(list(image_array))
        y_test.append(code[folder])
X_pred = list()

for folder in os.listdir(PredPath):
    Files = gb.glob(pathname = str(PredPath + "/" + folder + "/*jpg"))
    for file in Files:
        img = cv2.imread(file)
        image_array = cv2.resize(img , (s , s))
        X_Pred.append(list(image_array))
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)
X_pred  = np.array(X_pred)
Model = keras.models.Sequential([
        keras.layers.Conv2D(200 , kernel_size = (3 , 3) , activation = "relu" , input_shape = (s , s , 3)) ,
        keras.layers.Conv2D(150 , kernel_size = (3 , 3) , activation = "relu" ) ,
        keras.layers.MaxPool2D(4 , 4),
        keras.layers.Conv2D(120 , kernel_size = (3 , 3) , activation = "relu" ) ,
        keras.layers.Conv2D(80 , kernel_size = (3 , 3) , activation = "relu" ) ,
        keras.layers.Conv2D(50 , kernel_size = (3 , 3) , activation = "relu" ) ,
        keras.layers.MaxPool2D(4 , 4),
        keras.layers.Flatten() ,
        keras.layers.Dense(120 , activation = "relu"),
        keras.layers.Dense(100 , activation = "relu"),
        keras.layers.Dense(50  , activation = "relu"),
        keras.layers.Dropout(rate = 0.5) ,
        keras.layers.Dense(6)
    

])
Model.compile(optimizer = "adam" , loss =  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) ,  metrics = ["accuracy"])
Model.summary()
model = Model.fit(X_train ,y_train , epochs = 50  , verbose = 1 , batch_size = 64)
plt.plot(model.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
Model.evaluate(X_test ,  y_test , verbose= 1)
Model.save("Model1")