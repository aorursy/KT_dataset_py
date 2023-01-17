import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set();
df= pd.read_csv('../input/lubanumbatwo/bigml_59c28831336c6604c800002a.csv',sep=',')
df.head()
df.describe()
df.info()
sns.countplot(df['churn'])
sns.boxplot(df['total eve charge'])
sns.boxplot(df['account length'])
sns.boxplot(df['total day minutes'])
sns.boxplot(df['total night minutes'])
sns.boxplot(df['total intl minutes'])
#new data frame

ndf=pd.get_dummies(df,columns=['state','international plan','voice mail plan'])
ndf=ndf.drop('phone number',axis=1)
ndf=ndf.drop('area code', axis=1)
ndf['account length']=ndf['account length']/365
ndf['churn']=ndf['churn'].map({True:1,False:0})
ndf.head()
ndf.info()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()



X1=ndf.drop('churn',axis=1)

print(X1)
X=scaler.fit_transform(X1)

print(X)
y=ndf['churn']

print(y)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)

y_pred
from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
knn.score(X_valid, y_valid)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X, y, 

                         cv=kf, scoring='accuracy')

print(scores)

scores.mean()
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
pd.DataFrame(knn_grid.cv_results_).T
knn_grid.score(X_valid, y_valid)
best_knn = KNeighborsClassifier(n_neighbors=9)

y_pred = best_knn.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
results_df = pd.DataFrame(knn_grid.cv_results_)

plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])

plt.xlabel('n_neighbors')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
knn = KNeighborsClassifier(n_neighbors=9,weights='distance')

knn_params = {'p': np.linspace(1, 10,num=200)}

knn_grid = GridSearchCV(knn,

                        knn_params,

                        scoring='accuracy',

                        cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_params_
knn_grid.best_score_
knn_grid.score(X_valid, y_valid)
best_knn = KNeighborsClassifier(n_neighbors=9,p=1.949748743718593,weights='distance')

y_pred = best_knn.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
pd.DataFrame(knn_grid.cv_results_).T
results_df = pd.DataFrame(knn_grid.cv_results_)

plt.plot(results_df['param_p'], results_df['mean_test_score'])

plt.xlabel('p')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
from sklearn.neighbors import RadiusNeighborsClassifier

rnc=RadiusNeighborsClassifier(weights='distance')

rnc_params = {'radius': np.linspace(10, 1000,num=10)}

rnc_grid = GridSearchCV(rnc,

                        rnc_params,

                        scoring='accuracy',

                        cv=kf)

rnc_grid.fit(X_train, y_train)
rnc_grid.best_estimator_
rnc_grid.best_score_
rnc_grid.best_params_
pd.DataFrame(rnc_grid.cv_results_).T
results_df = pd.DataFrame(rnc_grid.cv_results_)

plt.plot(results_df['param_radius'], results_df['mean_test_score'])

plt.xlabel('radius_param')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
from sklearn.neighbors import RadiusNeighborsClassifier

rnc=RadiusNeighborsClassifier(weights='distance', radius=1000)

rnc_params = {'p': np.linspace(1, 10,num=10)}

rnc_grid = GridSearchCV(rnc,

                        rnc_params,

                        scoring='accuracy',

                        cv=kf)

rnc_grid.fit(X_train, y_train)
rnc_grid.best_estimator_
rnc_grid.best_params_
rnc_grid.best_score_
pd.DataFrame(rnc_grid.cv_results_).T
results_df = pd.DataFrame(rnc_grid.cv_results_)

plt.plot(results_df['param_p'], results_df['mean_test_score'])

plt.xlabel('p')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
from sklearn.neighbors import NearestCentroid

nc=NearestCentroid()

nc_params = {'metric':['chebyshev','manhattan','euclidean']}

nc_grid = GridSearchCV(nc,

                        nc_params,

                        scoring='accuracy',

                        cv=kf)

nc_grid.fit(X_train, y_train)
nc_grid.best_estimator_
nc_grid.best_score_
nc_grid.best_params_
pd.DataFrame(nc_grid.cv_results_).T
nc_grid.score(X_valid, y_valid)
best_nc = NearestCentroid(metric='euclidean')

y_pred = best_nc.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
pd.DataFrame(nc_grid.cv_results_)
results_df = pd.DataFrame(nc_grid.cv_results_)

plt.plot(results_df['param_metric'], results_df['mean_test_score'])

plt.xlabel('metric')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()