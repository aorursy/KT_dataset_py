import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

from scipy.ndimage.filters import gaussian_filter

%matplotlib inline

labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:5000,1:]

labels = labeled_images.iloc[0:5000,:1]



for i in range(images.shape[0]):

    img=images.iloc[i].as_matrix()

    img=img.reshape((28,28))

    img=gaussian_filter(img, sigma=1)

    img=img.reshape(784,1)

    images.iloc[i].as_matrix=img





train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)





i=50

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

img=gaussian_filter(img, sigma=1)

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
test_images[test_images<1]=0

test_images[test_images>=1]=1

train_images[train_images<1]=0

train_images[train_images>=1]=1

#train_images/=255

#test_images/=255



i=50

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)