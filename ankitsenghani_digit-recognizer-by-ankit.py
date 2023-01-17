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
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import svm

import numpy as np

%matplotlib inline
img = pd.read_csv('../input/train.csv')
images = img.iloc[:5000,1:]

labels = img.iloc[:5000,:1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
current_img = train_images.iloc[1].as_matrix()

current_img = current_img.reshape(28,28)

plt.imshow(current_img,cmap='gray')

plt.title(train_labels.iloc[1,0])

train_images[train_images>0] = 1

test_images[test_images>0] = 1
train_images.iloc[1].as_matrix()

current_img = train_images.iloc[1].as_matrix()

current_img = current_img.reshape(28,28)

plt.imshow(current_img,cmap='binary')

plt.title(train_labels.iloc[1,0])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_data = pd.read_csv('../input/test.csv')

test_data[test_data>0]=1

output_result = clf.predict(test_data)
output_result
of = pd.DataFrame(output_result)

of.index.name='ImageId'

of.index+=1

of.columns=['Label']

of.to_csv('results.csv', header=True)