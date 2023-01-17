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
import pandas as pd

import numpy as np
IrisD = pd.read_csv('/kaggle/input/irisdta/myIrisData.csv')
IrisD.head()
IrisD.shape
IrisD.isnull
IrisD.fillna(0,inplace = True)
import matplotlib.pyplot as plt
X = IrisD.drop('require',axis = 1)

y = IrisD['require']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
X_train.head()
y_train.head()
X_test.head()
y_test.head()
from sklearn import svm

Classifier = svm.SVC(kernel = 'linear',gamma = 'auto',C= 2)

Classifier.fit(X_train,y_train)
y_predict = Classifier.predict(X_test)

y_predict
from sklearn.metrics import classification_report
print(classification_report(y_predict,y_test))