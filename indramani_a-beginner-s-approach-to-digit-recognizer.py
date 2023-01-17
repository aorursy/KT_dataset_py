import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline
from sklearn import preprocessing
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
images = preprocessing.scale(images)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
test_images[test_images>0]=1
train_images[train_images>0]=1

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')
test_data= preprocessing.scale(test_data)
test_data[test_data>0]=1

results = clf.predict(test_data)
import numpy as np
results = pd.DataFrame(results)
results.index =results.index+1
results.columns =['Label']
results.to_csv('result.csv', index_label='ImageId')

