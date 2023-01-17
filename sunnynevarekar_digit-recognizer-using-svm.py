import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline

labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:20000, 1:]

labels = labeled_images.iloc[0:20000, 0:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

train_images = train_images/255

test_images = test_images/255

train_images.describe()
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')

test_data = test_data/255

results=clf.predict(test_data)
print(results.shape)

i = 21999

print(results[i])

img = test_data.iloc[i].as_matrix().reshape((28,28))*255

plt.imshow(img)
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True)