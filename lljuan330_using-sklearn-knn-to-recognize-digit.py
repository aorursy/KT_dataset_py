# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
label_train = train['label']
datas = train.drop(['label'],axis= 1)
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors= 20)
Knn.fit(datas,label_train)
result = Knn.predict(test)

