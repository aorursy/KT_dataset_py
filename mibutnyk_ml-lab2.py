import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import accuracy_score, mean_squared_error

from scipy.stats import normaltest



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv', sep=',')

df.head(10).T
df.info()
df.describe().T
df['quality'].hist(bins=11);
normaltest(df['quality'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



y = df['quality']

X = df.drop('quality', axis=1)

X_new = scaler.fit_transform(X)

print(X_new[:5, :5])
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_new,

                                                      y, 

                                                      test_size=0.2, 

                                                      random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)

knn.score(X_valid, y_valid)
accuracy_score(y_valid, y_pred)
knr = KNeighborsRegressor(n_neighbors=1)

knr.fit(X_train, y_train)

y1_pred = knr.predict(X_valid)

mean_squared_error(y_valid, y1_pred)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(knn, X_new, y,

                         cv=kf, scoring='accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)    
scores = cross_val_score(knn, X_new, y,

                         cv=kf, scoring='balanced_accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)
knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
grid_results = pd.DataFrame(knn_grid.cv_results_)

plt.plot(grid_results['param_n_neighbors'], grid_results['mean_test_score'])

plt.xlabel('n_neighbors')

plt.ylabel('knn_score')

plt.show()
knn2 = KNeighborsClassifier(n_neighbors=1, weights='distance')

knn2_params = {'p': np.linspace(1, 10, num=200, endpoint=True)}

knn2_grid = GridSearchCV(knn2, 

                        knn2_params, 

                        scoring='accuracy',

                        cv=kf)

knn2_grid.fit(X_train, y_train)
knn2_grid.best_params_
knn2_grid.best_score_
grid_results2 = pd.DataFrame(knn2_grid.cv_results_)

plt.plot(grid_results2['param_p'], grid_results2['mean_test_score'])

plt.xlabel('р')

plt.ylabel('knn_score')

plt.show()
from sklearn.neighbors import RadiusNeighborsClassifier

rnn = RadiusNeighborsClassifier(radius=5.0)

rnn.fit(X_train, y_train)

y2_pred = rnn.predict(X_valid)

rnn.score(X_valid, y_valid)
accuracy_score(y_valid, y2_pred)
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid()

nc.fit(X_train, y_train)

y3_pred = nc.predict(X_valid)

nc.score(X_valid, y_valid)
accuracy_score(y_valid, y3_pred)