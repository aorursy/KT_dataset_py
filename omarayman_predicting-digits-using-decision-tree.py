# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/train.csv").as_matrix()

clf = DecisionTreeClassifier()

train = data[0:21000,1:]

labels = data[0:21000,0]

clf.fit(train,labels)
xtest = data[21000:,1:]

xlabels = data[21000:,0]

p = clf.predict(xtest)

count = 0

for i in range(0,21000):

    count +=1 if p[i]==xlabels[i] else 0

print("Accuracy",(count/21000)*100 )
