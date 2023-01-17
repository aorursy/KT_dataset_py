import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import os
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
images = train.iloc[:,1:]
labels = train.iloc[:,:1]
train_X,val_X, train_y, val_y = train_test_split(images, labels, test_size=0.8, stratify = labels)
print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)
i=1
img = train_X.iloc[i]
img = img.values.reshape(28,28)
plt.imshow(img,cmap='Greys')
lr = LogisticRegression(n_jobs= 8 )
lr.fit(train_X, train_y)
val_predict = lr.predict(val_X)
(((val_predict == val_y['label'].values).sum()/val_predict.shape) * 100) [0] #84.01
train_X = train_X.clip(lower=0,upper=1)
val_X = val_X.clip(lower=0,upper=1)
test= test.clip(lower=0,upper=1)
i=55
img = train_X.iloc[i]
img = img.values.reshape(28,28)
plt.imshow(img,cmap='Greys')
lr = LogisticRegression()
lr.fit(train_X, train_y)
val_predict = lr.predict(val_X)
(((val_predict == val_y['label'].values).sum()/val_predict.shape) * 100) [0] #88.78
knnReg = KNeighborsClassifier(n_neighbors=10)
knnReg.fit(train_X, train_y)
val_predict_knn = knnReg.predict(val_X)
(((val_predict_knn == val_y['label'].values).sum()/val_predict_knn.shape) * 100) [0] #88.78
test_predict = knnReg.predict(test)
submission = pd.DataFrame(test_predict).reset_index()
submission.columns=['ImageId','Label']
submission.ImageId =  submission.ImageId + 1
submission.to_csv("knn_submission.csv",index=False)
