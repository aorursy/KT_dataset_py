#import several modules I used

import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline
train = pd.read_csv('../input/train.csv') 

test = pd.read_csv('../input/test.csv')
train.head(3)
test.head(3)
#pick up all images and all labels.

train_images = train.iloc[0:10000, 1:]

train_labels = train.iloc[0:10000, :1]



#If you decrease the nunber of digit you use, just type like  images = labeled_images.iloc[0:3000,1:]
train_images, test_images,train_labels, test_labels = train_test_split(train_images, train_labels, train_size=0.8, random_state=0)



#Try playing with the parameters of svm.

#SVC to see how the results change.

clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images, test_labels)
#Viewing an Image

img = train_images.iloc[1].as_matrix()

img = img.reshape((28,28))

plt.imshow(img, cmap = 'gray')

plt.title(train_labels.iloc[1,0])
#Examining the Pixel Values

plt.hist(train_images.iloc[1])
test_images[test_images>0] = 1

train_images[train_images>0] = 1



img = train_images.iloc[1].as_matrix().reshape((28,28))

plt.imshow(img, cmap='gray')

plt.title(train_labels.iloc[1])
plt.hist(train_images.iloc[1])
#Retraining our model!!

clf = svm.SVC()

clf.fit(train_images, train_labels)

clf.score(test_images,test_labels)
test_data = pd.read_csv('../input/test.csv')

test_data[test_data>0] = 1

results=clf.predict(test_data)
#submit

submission = pd.DataFrame({'Label': results})

submission.index += 1

submission.index.name = "ImageId"

submission.to_csv('submission.csv')