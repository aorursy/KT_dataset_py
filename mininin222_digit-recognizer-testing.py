import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



# Import the 3 dimensionality reduction methods

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.cross_validation import train_test_split



from sklearn.preprocessing import Imputer, Normalizer, scale

from sklearn.cross_validation import train_test_split, StratifiedKFold

from sklearn.feature_selection import RFECV
# Machine Learning Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import metrics
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/train.csv')
full_data = [train, test]
images = train.iloc[0:5000,1:]

labels = train.iloc[0:5000,:1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
images.head()

print (images[images>0].isnull().sum())
i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
i=2

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[5])

#From the histogram, notice that it's not entirely B&W pixels(0 or 1) but rather a 0-255 gray scale
images[images>0] = 255

img = images.iloc[1].as_matrix().reshape(28,28)

plt.imshow(img, cmap='binary')

plt.title(labels.iloc[1])
img = images.iloc[10].as_matrix().reshape(28,28)

plt.imshow(img, cmap='binary')

plt.title(labels.iloc[10])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test.iloc[0:10, :1]
iiclf.fit(train_images, train_labels.values.ravel())

predict = clf.predict(test.iloc[0:, 1:])
df = pd.DataFrame({'ImageId': range(1,test.index.size + 1), 'Label': predict})

print (df.head(10))

df.to_csv('digit_results.csv', index=False)

print ('Done!')
df.tail()
ddtest.index.size