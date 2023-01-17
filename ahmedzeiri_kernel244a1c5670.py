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

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')



#y_train = train.values[:,0]

#y_test = train.values[0:5000,0]

#x_test = test.values[:,1:]

#x_train = train.values[: , 1:]
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix



x = train.values[:,1:]

y = train.values[:,0]

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)



clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

y_pred = clf.predict(x_test)

acc1 = metrics.accuracy_score(y_test, y_pred)

matrix=confusion_matrix(y_test,y_pred)
acc1
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier

clf=RandomForestClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(x_train,y_train)

y_pred2=clf.predict(x_test)

acc2 = metrics.accuracy_score(y_test, y_pred2)

matrix2=confusion_matrix(y_test,y_pred2)