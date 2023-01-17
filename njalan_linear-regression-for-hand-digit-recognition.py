import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline
df = pd.read_csv('../input/train.csv')
labeled_images = df.iloc[0:5000,:]

isNull = labeled_images.isnull().sum().max()
print ("null entries : "+str(isNull))

labels = labeled_images['label']
images = labeled_images.drop(columns = ['label'])

train_images, test_images,train_labels, test_labels = train_test_split(images, labels)
print("shape of X Train :"+str(train_images.shape))
print("shape of X Test :"+str(test_images.shape))
print("shape of Y Train :"+str(train_labels.shape))
print("shape of Y Test :"+str(test_labels.shape))
index = 5
image = train_images.iloc[index].as_matrix()
image = image.reshape((28,28))
plt.imshow(image, cmap='gray')
plt.title(train_labels.iloc[index])
plt.hist(train_images.iloc[index])
clf = svm.SVC(gamma='auto')
clf.fit(train_images, train_labels.values.ravel())
acc = clf.score(test_images,test_labels)
print("accuracy: "+str(acc))
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[index].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[index])
plt.hist(train_images.iloc[index])
clf = svm.SVC(gamma='auto')
clf.fit(train_images, train_labels.values.ravel())
acc = clf.score(test_images,test_labels)
print("accuracy :"+str(acc))
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])
results
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns = ['Label']
df.to_csv('results.csv', header=True)