# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
path = "/kaggle/input/shapes/shapes"
print(os.listdir())
print(os.getcwd())

# Any results you write to the current directory are saved as output.

t_path = path +"/squares/drawing(1).png"  #saving the path for the example 1 in the triangles folder
print(os.path.exists(t_path))   #checking whether the path exists or not 
#Trying to load the image and display it 
import matplotlib.image as mpimg
t_path = path +"/squares/drawing(1).png"
img = mpimg.imread(t_path)
def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

gray = rgb2gray(img)
#print(type(gray))
#print(gray.shape)
imgplot = plt.imshow(gray,cmap = 'gray')
plt.show()     
img = np.ravel(gray)
#print(type(img))   #image type is numpy.ndarray
print(img.shape)  

df = pd.DataFrame([img])
df.head()   #this is the exmaple of 1 training example 
shapes = ['triangles','circles','squares']    #saving the list of the folders in shapes folder
path = "/kaggle/input/shapes/shapes/"         #path for the images folder

files = [] #for storing all the images path 
result = []
for shape in shapes:                                       #|
     new_path = path + shape                               #|
     for file in os.listdir(new_path):                     #|How can I make this code shorter?
        files.append(os.path.join(new_path,file))          #|
        result.append(shape)
#You can run this to check the values
#print(len(files))
#print(files[1])
#print(len(result))
images = [] #list for images  
for file in files:
    
    img = mpimg.imread(file)
    img = rgb2gray(img)
    img = np.ravel(img)
    images.append(img)
    
#print(len(images))
#print(len(images[0]))
df = pd.DataFrame(images)   #converting our images list into Pandas DataFrame
#a = np.array(images)
#df.head()
#df.shape
#df.describe()
#We have to add one column of labels of the images i.e. result column
df.loc[:, 'result'] = result

df.tail()
temp_df = df
temp_df.head()

#changing the result column data into numerical categories
temp_df.loc[:,'result'] = pd.factorize(temp_df.result)[0]
temp_df.head()
z_train = Counter(temp_df['result'])
z_train

sns.countplot(temp_df['result'])

from sklearn.utils import shuffle

#Now shuffling the rows in the dataframe
df = shuffle(temp_df)
df.head()
df.tail()
y = df.result
X = df.drop(['result'],axis = 1)
X = X.values.reshape(-1,28,28,1)
#applying one_hot encoding in the y
from keras.utils.np_utils import to_categorical #convert to one_hot encoding
y = to_categorical(y,num_classes = 3)
y.astype('int32')
from sklearn.model_selection import train_test_split  #for splitting our dataframe into training and test example
#Now splitting our dataset in train and test 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

y_test.astype('int32')
from keras.models import Sequential
from keras.layers import Dense, Flatten
model = Sequential()
model.add()
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.summary()
#compile model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#Train Model 
model.fit(X_train,y_train,batch_size=5,epochs=10)
z_train = Counter(y_train)
z_train
sns.countplot(y_train)
from sklearn import svm
clf = svm.SVC(gamma = 'auto')
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,prediction)
print(score)
from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
score = accuracy_score(y_test,prediction)
print(score)

from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()

clf.fit(X_train,y_train)

prediction = clf.predict(X_test)

score = accuracy_score(y_test,prediction)
print(score*100)