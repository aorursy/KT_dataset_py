# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv")
df.describe()
df.info()
df.head()
sns.catplot(x="churn", kind="count", palette="ch:.25", data=df)
df["area code"].value_counts()
df["state"].value_counts()
df_1 = df.drop(["state","phone number"], axis = 1)
df_2 = pd.concat([df_1, pd.get_dummies(df_1["area code"])], axis = 1)
df_2
y = df_2["churn"].map({False:0,True:1})
df_3 = df_2.drop(["area code", "churn"], axis = 1)
df_numeric = df_3.copy()
df_numeric["international plan"] = df_3["international plan"].map({"yes":1,"no":0})
df_numeric["voice mail plan"] = df_3["voice mail plan"].map({"yes":1,"no":0})
y
df_numeric
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df_numeric)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
X_train.shape
y_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_pred,y_test))

print(accuracy_score(y_pred,y_test))
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.model_selection import cross_val_score

print(cross_val_score(knn,X,y, cv = kf, scoring = "accuracy"))
from sklearn.model_selection import cross_val_score

print(cross_val_score(knn,X,y, cv = kf, scoring = "f1"))
from sklearn.model_selection import GridSearchCV

import numpy as np

params = {"n_neighbors" : np.arange(1, 51, 2)}

cv = GridSearchCV(knn, params, cv = kf, scoring="f1")

cv.fit(X,y)
cv.best_estimator_
cv.best_score_
cv.best_params_
cv_results = pd.DataFrame(cv.cv_results_)
cv_results
import matplotlib.pyplot as plt
plt.plot(cv_results["param_n_neighbors"],cv_results["mean_test_score"])
p_params = {"p": np.linspace(1,10,200)}

knn = KNeighborsClassifier(n_neighbors=3, weights = "distance", n_jobs = -1)

cv = GridSearchCV(knn, p_params, cv = kf, scoring="f1", verbose = 100)

cv.fit(X,y)
cv_results = pd.DataFrame(cv.cv_results_)

plt.plot(cv_results["param_p"],cv_results["mean_test_score"])
cv.best_estimator_
cv.best_score_
from sklearn.neighbors import RadiusNeighborsClassifier



radius_neighbors_classifier = RadiusNeighborsClassifier(radius=10)

radius_neighbors_classifier.fit(X_train, y_train)



radius_neighbors_classifier.score(X_test, y_test)
from sklearn.neighbors import NearestCentroid



nearest_centroid = NearestCentroid()

nearest_centroid.fit(X_train, y_train)

nearest_centroid.score(X_test, y_test)
from sklearn.svm import SVC
svc = SVC(kernel="rbf")

scores = cross_val_score(svc,X,y,cv=kf, scoring="f1")
scores.mean()
from sklearn.ensemble import RandomForestClassifier 
forest = RandomForestClassifier(n_estimators = 500, max_depth=5, random_state=42)
scores = cross_val_score(forest, X,y, cv=kf, scoring = "f1")
scores.mean()
corrmat = df_numeric.corr(method='spearman')

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Important variables correlation map", fontsize=15)

plt.show()
import xgboost as xgb



clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(df_numeric, y)

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax)

plt.show()
xgboostClassifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = cross_val_score(xgboostClassifier,X,y,cv=kf, scoring="f1")
scores.mean()
from sklearn.manifold import TSNE

X_1 = TSNE(n_components=2).fit_transform(df_numeric)

plt.scatter(X_1[:,0],X_1[:,1], c = y)
from sklearn.manifold import TSNE

X_1 = TSNE(n_components=2).fit_transform(X)

plt.scatter(X_1[:,0],X_1[:,1], c = y)
from sklearn.decomposition import PCA

X_2 = PCA(n_components=2).fit_transform(df_numeric)

plt.scatter(X_2[:,0],X_2[:,1], c = y)
from sklearn.decomposition import PCA

X_3 = PCA(n_components=2).fit_transform(X)

plt.scatter(X_3[:,0],X_3[:,1], c = y)
params = {"max_depth" : np.arange(3, 11), "n_estimators":[100,150,200,250,300,400]}

xgboostClassifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

cv = GridSearchCV(xgboostClassifier, params, cv = kf, scoring="f1", verbose = 100)

cv.fit(X,y)
cv_results = pd.DataFrame(cv.cv_results_)
plt.plot(cv_results["param_max_depth"],cv_results["mean_test_score"])
plt.plot(cv_results["param_n_estimators"],cv_results["mean_test_score"])
cv.best_estimator_
cv.best_score_
df_states = df.copy()
df_states = df_states.drop("phone number", axis = 1)
df_states
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_states['state'] = label_encoder.fit_transform(df_states['state'])

df_states['international plan'] = label_encoder.fit_transform(df_states['international plan'])

df_states['voice mail plan'] = label_encoder.fit_transform(df_states['voice mail plan'])
df_states = df_states.drop("churn", axis = 1)
X = scaler.fit_transform(df_states)
from sklearn.ensemble import GradientBoostingClassifier

gboost= GradientBoostingClassifier(max_depth=7, n_estimators = 100)

score = cross_val_score(gboost,X,y,cv = kf, scoring = "f1")
scores.mean()
kf = KFold(n_splits=10, shuffle=True, random_state=42)

gboost= GradientBoostingClassifier(max_depth=7, n_estimators = 100)

score = cross_val_score(gboost,X,y,cv = kf, scoring = "f1")

score.mean()
xgboost = xgb.XGBClassifier(max_depth=7, n_estimators=100, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

score = cross_val_score(gboost,X,y,cv = kf, scoring = "f1")

score.mean()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.metrics import classification_report

xgboost.fit(X_train,y_train)

y_pred = xgboost.predict(X_test)

print('Gradient Boosting Classifier:\n {}\n'.format(classification_report(y_test,y_pred)))