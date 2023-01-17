# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data

y = iris.target

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  

svc = svm.SVC(kernel='linear', C=15)

svc.fit(X_train, y_train)

predict= svc.predict(X_test)

predict
cnf_matrix = confusion_matrix(y_test, predict)

print(cnf_matrix)
accuracy=svc.score(X_test,y_test)

print (accuracy)
accuracy=svc.score(X_train,y_train)

accuracy