import pandas as pd

import numpy as np

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')



np.random.seed = 28

rand_image_index=np.random.randint(0,labeled_images.shape[0],5000)

images = labeled_images.iloc[rand_image_index,1:]

labels = labeled_images.iloc[rand_image_index,:1]



train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
clf = svm.SVC(C=10, gamma=0.01)

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_images[test_images>0]=1

train_images[train_images>0]=1



#img=train_images.iloc[23].as_matrix().reshape((28,28))

#plt.imshow(img,cmap='binary')

#plt.title(train_labels.iloc[23])
plt.hist(train_images.iloc[i])
clf = svm.SVC(C=10, gamma=0.01)

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')

test_data[test_data>0]=1

results=clf.predict(test_data)
test_data.shape

results.shape
df = pd.DataFrame({'ImageID': range(1, test_data.shape[0] + 1 ,1),

                  'Label': results})



#df = pd.DataFrame(results)

#df.index.name='ImageId'

#df.index+=1

#df.columns=['Label']
df.to_csv('results.csv', header=True)