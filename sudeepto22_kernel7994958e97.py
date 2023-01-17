import numpy as np

import pandas as pd
iris = pd.read_csv('../input/Iris.csv')

iris.head()
print(iris.shape)
iris.set_index('Id',inplace=True)
iris.columns
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y = iris['Species']
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



k_range = range(1,26)

score = {}



for k in k_range:

    knn = KNeighborsClassifier(k)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    score[k] = metrics.accuracy_score(y_test,y_pred)
score