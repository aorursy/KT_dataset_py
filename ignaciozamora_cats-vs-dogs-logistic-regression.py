import numpy as np
import pandas as pd
from zipfile import ZipFile
from keras.preprocessing.image import load_img 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
# Extract train and test set into folders

for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = "/kaggle/input/dogs-vs-cats/train.zip"
test = "/kaggle/input/dogs-vs-cats/test1.zip"

# Unzip files into working directories

if (not os.path.isdir('/train')):
    with ZipFile(train, 'r') as zip:  
        zip.extractall('/train')
        print('Train Files Extracted') 

if (not os.path.isdir('/test')):
    with ZipFile(test,'r') as zip:
        zip.extractall('/test')
        print('Test Files Extracted') 
# count the number of files in train and test set

train_loc = '/train/train'
test_loc = '/test/test1'

print("Training Set File Count:",len(os.listdir(train_loc)))
print("Training Set File Count:",len(os.listdir(test_loc)))
# this is the format of each file [class].[number].[file format] (dog.890.jpg)

train_set_files = os.listdir(train_loc)

# get classes and URL for each image in the traininig set.

y_train = [x.split('.')[0] for x in train_set_files]
X_train_url = ["/train/train/" + x  for x in train_set_files]

print(y_train[1:3],X_train_url[1:3])

# lets create a dataframe to hold the data so we can later use test-train split

data = pd.DataFrame(columns=['url','y'])
data.url = X_train_url
data.y = y_train

print(data.shape)
# lets print the first image to check how does it looks

img = load_img(X_train_url[0])
plt.imshow(img)

print("image size:",img.size)

# image size (target_size x target_size)
target_size = 64

# estimate the number of features of a flattened image
img = cv2.imread(X_train_url[0], cv2.COLOR_BGR2RGB) 
img = cv2.resize(img, (target_size, target_size))
img = img.reshape((target_size*target_size*3, 1))
features_count = len(img)
print("size of each image (array size):",features_count)
# lets create a test/train split out of the train set folder for validation (80/20)

X_train_url, X_test_url, y_train, y_test = train_test_split(data.url, data.y, test_size=0.2)
# iterate over each image and lets convert it into a numpy array
# this might take a while....

X_train = np.ones((1,features_count), int)

for path in X_train_url: 
    #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    #img = cv2.resize(img, (target_size, target_size)).flatten()
    img = cv2.imread(path, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (target_size, target_size))
    img = img.reshape((target_size*target_size*3, 1))
    X_train = np.append(X_train,img.T, axis = 0)
    
print("X_train created")
X_test = np.ones((1,features_count), int)
    
for path in X_test_url: 
    #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    #img = cv2.resize(img, (target_size, target_size)).flatten() 
    img = cv2.imread(path, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (target_size, target_size))
    img = img.reshape((target_size*target_size*3, 1))
    X_test = np.append(X_test,img.T, axis = 0)
    
print("X_test created")
# remove extra ones
X_train = np.delete(X_train, 0, 0)
X_test = np.delete(X_test, 0, 0)
from sklearn.linear_model import LogisticRegression

X_train_scale = X_train/255
X_test_scale = X_test/255

model = LogisticRegression(max_iter=10000).fit(X_train_scale, y_train)

print(model)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

y_test_predict = model.predict(X_test_scale)

# model accuracy
print("model acc:",accuracy_score(y_test, y_test_predict))

# prediction report
print(classification_report(y_test, y_test_predict))