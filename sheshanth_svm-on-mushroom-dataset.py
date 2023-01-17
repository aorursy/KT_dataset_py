import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
df_mushroom = pd.read_csv("../input/mushrooms.csv")
df_mushroom.head()
df_mushroom.describe().transpose()
from sklearn.preprocessing import LabelEncoder
df_mushroom = df_mushroom.apply(LabelEncoder().fit_transform)
df_mushroom.head()
X = df_mushroom.iloc[:,1:]
y = df_mushroom.iloc[:, 0]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
from sklearn.svm import SVC
svc_model = SVC(C=1, gamma='auto')
svc_model.fit(X=x_train, y=y_train)
y_pred = svc_model.predict(x_test)
from sklearn import metrics
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
from sklearn.model_selection import cross_val_score
cv_model = cross_val_score(svc_model, X, y, cv=10)
cv_model
from sklearn.model_selection import GridSearchCV
params = {'C':[1,10,100, 1000, 10000]}
svc_model = SVC()
model_grid_search = GridSearchCV(svc_model, cv=10, param_grid=params, scoring='accuracy')
model_grid_search.fit(X, y)
grid_search_results = pd.DataFrame(model_grid_search.cv_results_)
grid_search_results.transpose()
params = {'C':[1,10,100, 1000, 10000],
         'gamma':[0.1, 0.001, 0.0001, 0.00001]}
svc_model = SVC()
grid_search__model = GridSearchCV(svc_model, param_grid=params, cv=10, scoring='accuracy')
grid_search_score = grid_search__model.fit(X, y)
pd.DataFrame(grid_search__model.cv_results_).transpose()
