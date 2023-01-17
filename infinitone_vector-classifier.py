# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys, os

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# take the first 5000 images, separate them as labels / images

label_images = pd.read_csv('../input/train.csv')

images = label_images.iloc[0:5000,1:]

labels = label_images.iloc[0:5000,:1]



train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
train_images.info()

print('-'*100)

train_images.sample(2)
print(train_images.shape)

print(valid_images.shape)
train_images.describe()
i=3

s=train_images.iloc[i].as_matrix()

s=s.reshape((28,28))

#plt.imshow(s,cmap='gray')

#plt.title(train_labels.iloc[i,0])
#plt.hist(train_images.iloc[i])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(valid_images, valid_labels) # will give 10% accuracy, this is terribly.. need to improve it
# make images black and white

valid_images[valid_images>0]=1

train_images[train_images>0]=1



s=train_images.iloc[i].as_matrix().reshape((28,28))

plt.imshow(s,cmap='binary')

plt.title(train_labels.iloc[i])
plt.hist(train_images.iloc[i])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(valid_images, valid_labels)
test_data=pd.read_csv('../input/test.csv')

test_data[test_data>0]=1

results=clf.predict(test_data[0:5000])
results
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True)