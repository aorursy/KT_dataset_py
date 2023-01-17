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
iris_d = pd.read_csv("../input/Iris.csv")
print(iris_d.describe())
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

X = iris_d[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = iris_d["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = SVC()
clf.fit(X_train,y_train)
#print(clf.predict(X_test))
#print(y_test)
correct = 0
total = 0
for i,label in enumerate(clf.predict(X_test)):
    total += 1
    if(y_test.iloc[i] == label):
        correct += 1
print("Accuracy: "+str((correct*100)/total)+"%")