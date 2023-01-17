# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

import numpy as np

import matplotlib.pyplot as plt
handwritten_digits = datasets.load_digits()
X = handwritten_digits['data']

y = handwritten_digits['target']

X_train1,X_test,y_train1,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
X_train,X_valid,y_train,y_valid = train_test_split(X_train1,y_train1,test_size=0.3,random_state=10)
X_train.shape,y_train.shape
X_valid.shape,y_valid.shape
krange = range(2,10)

krange
k_accuracy = {}

for k in krange:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    score = knn.score(X_valid,y_valid)

    k_accuracy[k]=score

print(k_accuracy)
k=0

for kval in k_accuracy.keys():

    if k>0:

        if k_accuracy[kval] > k_accuracy[k]:

            k = kval

    else:

        k = kval

k

            
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train1,y_train1)

for i in np.random.randint(0,high=len(y_test),size=(5,)):

    image = X_test[i]

    predictions = knn.predict([image])

    imgdata = np.array(image,dtype='float')

    pixels = imgdata.reshape((8,8))

    plt.imshow(pixels,cmap='gray')

    plt.show()

    print(predictions)

    


