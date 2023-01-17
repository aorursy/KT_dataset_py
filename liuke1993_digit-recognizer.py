# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline





# read data and split it into training data and validation data

labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:5000,1:]

labels = labeled_images.iloc[0:5000,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)



# visualize a sample from training data

i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_images[test_images>0]=1

train_images[train_images>0]=1



img=train_images.iloc[i].as_matrix().reshape((28,28))

plt.imshow(img,cmap='binary')

plt.title(train_labels.iloc[i])

clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')

test_data[test_data>0]=1

results=clf.predict(test_data[0:5000])

results
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True)

print(check_output(["ls"]).decode("utf8"))