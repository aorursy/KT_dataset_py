# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #mat



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import the datasets with pandas

mydataset = pd.read_csv('../input/Iris.csv')
X = mydataset.iloc[:,[1,2,3,4]].values



X
y = mydataset['Species']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=0)



print('There are {} samples in the Training Set and {} samples in the Test Set'.format(X_train.shape[0], X_test.shape[0]))

from sklearn.svm import SVC



model = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

y_predicted_train = model.predict(X_train)







y_predicted
y_test
model.score(X_train, y_train)



model.score(X_test, y_test)