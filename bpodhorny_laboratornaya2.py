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
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("../input/../input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv", sep=",")

df.T
df.info()
df.groupby("churn")["phone number"].count().plot(kind='bar') 
df["state"].value_counts()
df["state"].value_counts().shape[0]
df["area code"].value_counts()
df1 = df.drop(["state", "phone number"], axis=1)

df2 = df1.copy()



df2["area code"] = df['area code'].map({408:0,415:1,510:2})

df2["international plan"] = df['international plan'].map({"no":0,"yes":1})

df2["voice mail plan"] = df['voice mail plan'].map({"no":0,"yes":1})

df2["churn"] = df['churn'].map({False:0,True:1})



df2
df_without_target = df2.drop(["churn"], axis=1)

df_without_target 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df_without_target )

X
from sklearn.model_selection import train_test_split



y = df2["churn"]



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state=12)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)

y_pred
knn.score(X_valid, y_valid)
from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X, y, 

                         cv=kf, scoring='accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid_cv = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=kf)

knn_grid_cv.fit(X_train, y_train)
knn_grid_cv.best_estimator_
knn_grid_cv.best_score_
knn_grid_cv.best_params_
cv_results_df = pd.DataFrame(knn_grid_cv.cv_results_)

cv_results_df
plt.plot(cv_results_df['param_n_neighbors'], cv_results_df['mean_test_score'])

plt.xlabel('neighbours')

plt.ylabel('test_score')
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn_f1 = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn_f1, X, y, 

                         cv=kf, scoring='f1')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid_cv_f1 = GridSearchCV(knn, 

                        knn_params, 

                        scoring='f1',

                        cv=kf)

knn_grid_cv_f1.fit(X_train, y_train)
knn_grid_cv_f1.best_estimator_
knn_grid_cv_f1.best_score_
knn_grid_cv_f1.best_params_
cv_f1_results_df = pd.DataFrame(knn_grid_cv_f1.cv_results_)

cv_f1_results_df
plt.plot(cv_f1_results_df['param_n_neighbors'], cv_f1_results_df['mean_test_score'])

plt.xlabel('neighbours')

plt.ylabel('test_score')
knn_weights = KNeighborsClassifier(n_neighbors=7, weights = "distance")

knn_weights.fit(X_train, y_train)
p_weights = {"p": np.linspace(1,10, 200)}

knn_weights_cv = GridSearchCV(knn_weights, p_weights, scoring="accuracy", cv = kf)

knn_weights_cv.fit(X_train, y_train)
knn_weights_cv.best_estimator_
knn_weights_cv.best_score_
knn_weights_cv_results = pd.DataFrame(knn_weights_cv.cv_results_)

knn_weights_cv_results
plt.plot(knn_weights_cv_results["param_p"],knn_weights_cv_results["mean_test_score"])
from sklearn.neighbors import RadiusNeighborsClassifier



radiusnbrclsf = RadiusNeighborsClassifier(radius=5)

radiusnbrclsf.fit(X_train, y_train)



y_pred = radiusnbrclsf.predict(X_valid)

radiusnbrclsf.score(X_valid, y_valid)
from sklearn.neighbors import RadiusNeighborsRegressor



radiusnbrregsor = RadiusNeighborsRegressor(radius=5)

radiusnbrregsor.fit(X_train, y_train)



y_pred = radiusnbrregsor.predict(X_valid)

radiusnbrregsor.score(X_valid, y_valid)
from sklearn.neighbors import NearestCentroid



ncentroid = NearestCentroid()

ncentroid.fit(X_train, y_train)



y_pred = ncentroid.predict(X_valid)

ncentroid.score(X_valid, y_valid)