import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline

#Loading the data.Just 1000 images for a short test
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:10000,1:]
labels = labeled_images.iloc[0:10000,:1]
#Setup of the necessary variables for training an testing the network
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


#black and white
test_images[test_images>0]=1
train_images[train_images>0]=1
i=23
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

#train. The result should be better
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

#submission
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),test_images],axis = 1)
submission.to_csv("submission.csv",index=False)

