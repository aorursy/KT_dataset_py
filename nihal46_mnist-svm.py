# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn import datasets,svm,metrics
df=pd.read_csv("../input/mnist_train.csv")

dft=pd.read_csv("../input/mnist_test.csv")
train=df.values

test=dft.values



train_data=train[:,1:]

train_labels=train[:,0]



test_data=test[:,1:]

test_labels=test[:,0]



train_data=train_data/255

test_data=test_data/255
classifier=svm.SVC(C=200,kernel='rbf',gamma=0.01,probability=False)

classifier.fit(train_data,train_labels)
predicted=classifier.predict(test_data)
print("Classification report for classifier :\n%s\n" % (metrics.classification_report(test_labels, predicted)))