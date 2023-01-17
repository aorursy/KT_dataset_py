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
import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')

train_data.head(5)

train_data.shape

train_images = train_data.iloc[0:3000,1:]

train_lables = train_data.iloc[0:3000,0:1]

type(train_data.iloc[0,1:].as_matrix())

train_data.iloc[0,1:].as_matrix().shape

type(train_data.iloc[0,1:])
tstImg = train_images.iloc[3].as_matrix()

tstImg = tstImg.reshape(28,28)

plt.imshow(tstImg,cmap='gray')

plt.title(train_lables.iloc[3,0])
from sklearn import svm

from sklearn import metrics



svm_clf = svm.SVC(gamma=00.1,kernel='rbf')

svm_clf.fit(train_images,train_lables[:,0])

metrics.score()