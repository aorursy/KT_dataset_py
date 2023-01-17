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

import numpy

import xgboost

from sklearn import cross_validation

from sklearn.metrics import accuracy_score
dataset = np.loadtxt('../input/pima-indians-diabetes.data.csv', delimiter=",")



X = dataset[:,0:8]

Y = dataset[:,8]
#split data into train and stest sets

seed = 7

test_size = 0.33

X_train , X_test, y_train, y_test= cross_validation.train_test_split(X,Y,test_size=test_size,random_state=seed)
model = xgboost.XGBClassifier()

model.fit(X_train,y_train)
print(model)
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))