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
from math import sqrt

import pandas as pd
dataset = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

dataset_test=pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
data_test=dataset_test.iloc[:,1:].values

data_test
dataset.head()
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)

classifier.fit(x_train, y_train)
import numpy as np

y_pred = classifier.predict(x_test)

pred = classifier.predict(data_test)

#np.shape(data_test)

np.shape(x_test)
dataset_test
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
accuracy=cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]

accuracy*=100/len(y_test)
accuracy
classifier.predict(data_test)