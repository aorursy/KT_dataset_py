import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn import preprocessing

%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')

for ii in range(2, 5):

    images = labeled_images.iloc[0:5000,ii].reshape(-1, 1)

    le = preprocessing.LabelEncoder()

    le.fit(images)

    images = le.transform(images).reshape(-1, 1)

    labeled_images.iloc[0:5000,ii]= images

    



    

images = labeled_images.iloc[0:5000,2:5]    

labels = labeled_images.iloc[0:5000,1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

print(train_images)

clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
results=clf.predict(images[0:12])

results
labeled_images = pd.read_csv('../input/test.csv')

for ii in range(1, 4):

    images = labeled_images.iloc[0:5000,ii].reshape(-1, 1)

    le = preprocessing.LabelEncoder()

    le.fit(images)

    images = le.transform(images).reshape(-1, 1)

    labeled_images.iloc[0:5000,ii]= images

    



    

images = labeled_images.iloc[0:5000,1:4]    

labels = labeled_images.iloc[0:5000,1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

clf.score(test_images,test_labels)