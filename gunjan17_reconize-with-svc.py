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
images_train = data_train.iloc[0:12000,1:]

labels_train = data_train.iloc[0:12000,:1]
images_train.head()
i=3

img=images_train.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(labels_train.iloc[i,0])
b= np.arange(1,28001)
sv = svm.SVC()
sv.fit(images_train,labels_train.values.ravel())
sv.score(images_train,labels_train.values.ravel())
presv=sv.predict(data_test)
presv = np.array(presv)
submission = pd.DataFrame({

        "ImageId": b,

        "Label": presv

    })



submission.to_csv('mnistsv.csv',index=False)