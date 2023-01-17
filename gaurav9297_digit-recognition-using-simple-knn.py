# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

%matplotlib inline
dataset=pd.read_csv('../input/train.csv')

images=dataset.iloc[0:5000,1:]

labels=dataset.iloc[0:5000,:1]

train_images,test_images,train_labels,test_labels=train_test_split(images,labels,random_state=2,train_size=0.8)
i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape(28,28)

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i])

plt.hist(train_images.iloc[i])
from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier()

clf.fit(train_images,train_labels.values.ravel())
predictions=clf.predict(test_images)
print(predictions)
from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels,predictions))
testd=pd.read_csv('../input/test.csv')

result=clf.predict(testd)

print(result)
df=pd.DataFrame(result)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv',header=True)