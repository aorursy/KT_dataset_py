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

from sklearn import svm

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
pd.set_option('display.max_columns',5000)
data_train.head()
data_test.head()
data_train.describe()
images_train = data_train.iloc[0:20000,1:]

labels_train = data_train.iloc[0:20000,:1]
images_train.head()
labels_train.head()
i=3

img=images_train.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(labels_train.iloc[i,0])
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(images_train,labels_train.values.ravel())
lr.score(images_train,labels_train.values.ravel())
pre=lr.predict(data_test)
i=2351

img1=data_test.iloc[i].as_matrix()

img1=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(pre[i])
b= np.arange(1,28001)

len(b)
pre = np.array(pre)
len(pre)
submission = pd.DataFrame({

        "ImageId": b,

        "Label": pre

    })



submission.to_csv('mnist.csv',index=False)