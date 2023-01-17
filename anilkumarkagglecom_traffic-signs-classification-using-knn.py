# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.
#reading the number of classes from meta file
meta_data = pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Meta.csv')
meta_shape = meta_data.shape
no_classes = meta_shape[0]
# Resizing all images and converting then to numpy array and adding labels
#let's convert then to 40x40 size
#the images are in RGB format. so the number od channels for each image is 3
#the 
import cv2
train_data=[]
train_labels=[]

side = 20
channels = 3


for c in range(no_classes) :
    path = "../input/gtsrb-german-traffic-sign/train/{0}/".format(c)
    files = os.listdir(path)
    for file in files:
        train_image = cv2.imread(path+file)
        image_resized = cv2.resize(train_image, (side, side), interpolation = cv2.INTER_AREA)
        train_data.append(np.array(image_resized))
        train_labels.append(c)
data = np.array(train_data)
data = data.reshape((data.shape[0], 20*20*3))
data_scaled = data.astype(float)/255
labels = np.array(train_labels)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data_scaled, labels, test_size=0.25, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))
from sklearn import metrics
print(metrics.accuracy_score(y_pred, y_val))
#now predicting the model with the test data:
from pylab import *
from skimage import io
test_path = "../input/gtsrb-german-traffic-sign/test/"
test_files = os.listdir(test_path)
for i in range(15):
    img = io.imread(test_path+test_files[i])
    figure(i)
    io.imshow(img)
from PIL import Image
y_test=pd.read_csv("../input/gtsrb-german-traffic-sign/Test.csv")
labels=y_test['Path']
y_test=y_test['ClassId'].values

data=[]

for f in labels:
    image=cv2.imread('../input/gtsrb-german-traffic-sign/test/'+f.replace('Test/', ''))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((20, 20))
    data.append(np.array(size_image))

X_test=np.array(data)
X_test = X_test.astype('float32')/255
X_test = X_test.reshape((X_test.shape[0], 20*20*3))
pred = model.predict(X_test)
print(metrics.accuracy_score(pred, y_test)) 
print(classification_report(y_test, pred))
test_path = "../input/gtsrb-german-traffic-sign/test/"
test_files = os.listdir(test_path)
for i in range(15):
    img = io.imread(test_path+test_files[i])
    figure(i)
    io.imshow(img)
    print('original: ', y_test[i], ' predicted: ', pred[i])
