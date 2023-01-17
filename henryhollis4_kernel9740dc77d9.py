from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


df = pd.read_csv('../input/train.csv')
#extracting the target values
y = df.iloc[0:, 55].values

X = df.iloc[0:, 0:55].values 

dfTest = pd.read_csv('../input/test.csv')
X_test = dfTest.iloc[0:, 0:55].values
TestID = dfTest.iloc[0:, 0].values



knn = KNeighborsClassifier(n_neighbors=5, p=1, metric='minkowski')
knn.fit(X, y)

y_pred = knn.predict(X_test)


dfOut = pd.DataFrame({"Id" : TestID, "Cover_Type" : y_pred})
dfOut.to_csv('submission1.csv', index=False)

