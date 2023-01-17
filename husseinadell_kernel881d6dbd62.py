# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

import csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_data = np.array(train.iloc[0: , 1:])

train_label = np.array(train.iloc[0: , 0])

print(train_label)
neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(train_data, train_label) 



test_data = np.array(test.iloc[0: , 0:])

test_label = neigh.predict(test_data)
print(test_label)
temp = np.zeros((28001,2)).astype(str)

temp[0,0] = str(temp[0,0])

temp[0,0] = 'ImageId'

temp[0,1] = 'Label'

for i in range (1,28001):

    temp[i,0] = i

temp[1:,1] = test_label



print(temp)
ocr = open('ocr.csv' , 'w')

wr = csv.writer(ocr, quoting=csv.QUOTE_ALL)

for i in range (len(temp)):

    if i == 0:

        wr.writerow(temp[i])

    else:

        wr.writerow([int(float(temp[i][0])),int(float(temp[i][1]))])





ocr.close()