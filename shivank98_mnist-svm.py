import pandas as pd

import matplotlib.pyplot as plt,matplotlib.image as mping

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline

train = pd.read_csv('../input/train.csv')

images = train.iloc[:, 1:]

labels = train.iloc[:, :1]



train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size = 0.8, random_state = 0)
i = 8

img = train_images.iloc[i].as_matrix()

img = img.reshape((28,28))

plt.imshow(img, cmap= 'gray')

plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
test_images[test_images>0]=1

train_images[train_images>0]=1

clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)

test = pd.read_csv('../input/test.csv')

test[test > 0] = 1



results = clf.predict(test[:])
results
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']



df.to_csv('results_finl.csv', header=True)


