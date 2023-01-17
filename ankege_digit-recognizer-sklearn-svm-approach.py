import pandas as pd

import numpy

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:5000,1:]

labels = labeled_images.iloc[0:5000,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
img=train_images.iloc[1].to_numpy()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[1,0])
plt.hist(train_images.iloc[1])
Cset = [0.33, 1, 3, 9, 27]

bestscore = 0

for i in Cset:

    clf = svm.SVC(C=i)

    clf.fit(train_images, train_labels.values.ravel())

    score = clf.score(test_images,test_labels)

    if score > bestscore:

        bestscore = score

        bestclf = clf

    print(i, score, bestscore)
test_data=pd.read_csv('../input/test.csv')

results=bestclf.predict(test_data)
results
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True)