import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')
test.head()
images=train.iloc[:5000,1:]
labels=train.iloc[:5000,:1]
images.head(20)
labels.head()
train_images,test_images,train_labels,test_labels=train_test_split(images,labels,train_size=0.8,random_state=0)
i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape(28,28)
plt.imshow(img,cmap='gray')
len(images.columns)
train_images.iloc[i].head()
plt.hist(train_images.iloc[i])
clf=svm.SVC()
clf.fit(train_images,train_labels.values.ravel())
clf.score(test_images,test_labels)
test_images[test_images>0]=1
train_images[train_images>0]=1
img=train_images.iloc[i].as_matrix().reshape(28,28)

plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
plt.hist(train_images.iloc[i])
clf=svm.SVC()
clf.fit(train_images,train_labels.values.ravel())
clf.score(test_images,test_labels)
test[test>0]=1
result=clf.predict(test)
result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("submission.csv",index=False)
