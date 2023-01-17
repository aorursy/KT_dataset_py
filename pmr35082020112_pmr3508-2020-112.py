import pandas as pd

import numpy as np



from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/adult-pmr3508/train_data.csv', index_col=0)

test = pd.read_csv('../input/adult-pmr3508/test_data.csv', index_col=0)

y_test = pd.read_csv('../input/adult-pmr3508/sample_submission.csv', index_col=0)

train.shape
test.head()
nTrain = train.dropna()

nTest = test.dropna()

y_test = y_test.dropna()

print(y_test.shape, nTest.shape)
X_train = nTrain[["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]]

y_train = nTrain.income



X_test = nTest[["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]]

X_train.head()
from pandas.plotting import scatter_matrix

sm = scatter_matrix(X_train, figsize=(10, 10))
X_train.corr()
sc_X = StandardScaler()

X_Train = sc_X.fit_transform(X_train)

X_Test = sc_X.transform(X_test)
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(15,40)}
knn_gscv = GridSearchCV(knn, param_grid, cv=10)
knn_gscv.fit(X_Train, y_train)
print(knn_gscv.best_params_, knn_gscv.best_score_)
knn = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])

knn.fit(X_Train, y_train)

y_Test_Pred = knn.predict(X_Test)
submission = pd.DataFrame()
submission[0] = nTest.index

submission[1] = y_Test_Pred

submission.columns = ["Id", "Income"]

submission.head()
submission.to_csv("submission.csv", index = False)