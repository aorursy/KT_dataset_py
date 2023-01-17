import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
cover = pd.read_csv("../input/covtype.csv")

cover      #liat datasat

cover.info()    #liat datatype
cover["Cover_Type"]

cover["Cover_Type"].describe() #summary dataset
X = cover[cover.columns[0:54]]

Y = cover["Cover_Type"]
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel



clf = ExtraTreesClassifier()

clf = clf.fit(X, Y)

model = SelectFromModel(clf, prefit=True)

New_features = model.transform(X)

model.get_support()

New_features.shape
New_features
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(New_features, Y, test_size=0.30, random_state=42)

X_train

X_train.shape

X_test.shape
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=3, weights='uniform')

model.fit(X_train, y_train)

model.score(X_train,y_train)  #nilai akurasi, dievaluasi ke x_train dan y_train
from sklearn import metrics



y_pred = model.predict(X_test)

y_pred.shape

acc_test = metrics.accuracy_score(y_test,y_pred)

#acc_test