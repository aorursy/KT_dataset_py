# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
#load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)

# now we gonna load the second image, reshape it as matrix than display it

i = 1
img = train_images.iloc[i].values
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
print(type(images.iloc))
print(np.unique(train_labels.iloc[:, 0].values))
count_label = len(train_labels)
print(count_label)

print(train_labels.head(45))
index = [12, 4, 18, 2, 42, 9, 1, 14, 11, 7]

for i in index:
    plt.figure()
    img = train_images.iloc[i].values
    img = img.reshape((28,28))
    plt.imshow(img,cmap = 'gray')
    plt.title(train_labels.iloc[i,0])

#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
#train_images.iloc[i].describe()
#print((type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(len(data3))
plt.hist(data3)

index = [12, 4, 18, 2, 42, 9, 1, 14, 11, 7]

for i in index:
    img = train_images.iloc[i].values
    plt.figure()
    plt.hist(img)
    plt.title(str(train_labels.iloc[i,0]))
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 0)
model.fit(train_images, train_labels)
predictions = model.predict(test_images)
MAE = mean_absolute_error(test_labels, predictions)
print(MAE)
#print(train_labels.values.ravel())
#print(np.unique(test_labels)) # to see class number
index = [12, 4, 18, 2, 42, 9, 1, 14, 11, 7]
for i in index:
    test_images[test_images>0]=1
    train_images[train_images>0]=1
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.figure()
    plt.imshow(img,cmap='binary')
    plt.title("Kelas ke - " + str(train_labels.iloc[i,0]))
# now plot again the histogram
#plt.hist(train_images.iloc[i])
index = [12, 4, 18, 2, 42, 9, 1, 14, 11, 7]
for i in index:
    img = train_images.iloc[i].values
    plt.figure()
    plt.hist(img)
    plt.title("Kelas ke - " + str(train_labels.iloc[i,0]))