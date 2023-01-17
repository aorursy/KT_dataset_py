# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



csv_data = pd.read_csv('../input/train.csv')

train_label = csv_data.iloc[1:,:1].values.ravel()

train_img = csv_data.iloc[1:,1:]



test_csv_data = pd.read_csv("../input/test.csv")

test_data = test_csv_data.values#.iloc[0:,:]

#print(test_data.shape)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(train_img, train_label)



y = knn.predict(test_data)

pd.DataFrame({"ImageId":list(range(1,len(y)+1)),"Label":y}).to_csv("xx.csv", index=False,header=True)