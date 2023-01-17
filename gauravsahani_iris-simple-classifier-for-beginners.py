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
import sklearn
df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df.tail()


df.shape

df.isnull().sum()
X=df[['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y=df[['Species']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
predict1=knn.predict(X_test)
from sklearn import metrics
print("The accuracy of the KNN Classifier is: ",metrics.accuracy_score(predict1, y_test))
from sklearn import svm
SVM=svm.SVC()
SVM.fit(X_train, y_train)
predict2=SVM.predict(X_test)
print("The accuracy of the SVM Classifier is: ",metrics.accuracy_score(predict2, y_test))
from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
LR.fit(X_train, y_train)
predict3 = LR.predict(X_test)
print("The accuracy of the LogisticRegression Classifier is: ",metrics.accuracy_score(predict3, y_test))