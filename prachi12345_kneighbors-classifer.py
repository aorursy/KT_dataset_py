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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
df=pd.read_csv("../input/auto-mpg.csv")
df.head(10)
df.info()
df=df.drop(['horsepower'],1)
df
neighbors=KNeighborsClassifier(n_neighbors=3)
x=df.values[:,0:5]
org=df.values[:,6]
org = org.astype('int')
x
trainX, testX, trainY, testY = train_test_split( x, org, test_size = 0.3)
neighbors.fit(trainX, trainY)
pred=neighbors.predict(testX)
pred
testX
