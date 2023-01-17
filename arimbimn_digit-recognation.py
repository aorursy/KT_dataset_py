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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
#%matplotlib inline
#print(type(plt))
#print(type(mpimg))
#print(type(train_test_split))
#print(type(svm))
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)

#print(train_images, test_images,train_labels, test_labels)
i=1
img=train_images.iloc[i].values
#print(img)
img=img.reshape((28,28))
#print(img)
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
#print(plt.imshow)
print(img)
#Answer Q2:
print (np.unique(train_labels))
#for i in range(len(train_labels)): 
    #print(i, train_labels.iloc[i, 0]) untuk mencari indeks labels
#labels = 0,1,2,3,4,5,6,7,8,9 | index = 12,4,18,2,42,9,1,0,13,62
array = [12,4,18,2,42,9,1,0,13,62]
for i in array:
    plt.figure()
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])
    plt.show()
    plt.close()


#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
#Answer Q3:
#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
array = [12,4,18,2,42,9,1,0,13,62]
for i in array:
    img = train_images.iloc[i].values
    plt.hist(img)
    plt.title(train_labels.iloc[i,0])
    plt.show()
    plt.close()
#contoh
#print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(len(data3))
plt.hist(data3)
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Answer Q4: 
# Menggunakan SVM.score()
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
print("clf.score = ", clf.score(test_images,test_labels))
# Menggunakan MAE
data_model = DecisionTreeRegressor(random_state= 0)
data_model.fit(train_images,train_labels)
predictions = data_model.predict(test_images)
mae = mean_absolute_error(test_labels, predictions)
print("MAE = ", mae)
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number
#test_images[test_images>0]=1
#train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
print(img)
# now plot again the histogram
print(plt.hist(train_images.iloc[i]))
print(plt.title(train_labels.iloc[i]))
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#Answer Q5:
print (np.unique(train_labels))
#print(train_labels)
#label = 0,1,2,3,4,5,6,7,8,9 | index = 12,4,18,2,42,9,1,0,13,62
array = [12,4,18,2,42,9,1,0,13,62]
for i in array:
    plt.figure()
    img=train_images.iloc[i].values.reshape((28,28))
    plt.imshow(img,cmap='binary')
    plt.title(train_labels.iloc[i])
    plt.show()
    plt.close()
# Test again to data test
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])
# separate code section to view the results
print(results)
print(len(results))
# dump the results to 'results.csv'
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
#check if the file created successfully
print(os.listdir("."))
