# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv')
df.head()
df.describe().T
target = df['shares']
target
snsplot = sns.kdeplot(df['shares'], shade=True)
fig = snsplot.get_figure()
df['shares_log'] = np.log(df['shares'])
snsplot = sns.kdeplot(df['shares_log'], shade=True)
fig = snsplot.get_figure()
stat, p = shapiro(df['shares_log'] )
p
df.info()
df1 = df.drop('url', axis = 1)
df1
df1 = df1.drop('shares', axis=1)
X = df1.drop('shares_log', axis=1)
y = df1['shares_log'] 
X.head()
scaler = StandardScaler()
X_new = scaler.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, test_size = 0.25, random_state = 19)
knn = KNeighborsRegressor(n_neighbors = 50)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)
y_pred
knn.score(X_valid, y_valid)
mean_squared_error(y_valid, y_pred)
kf = KFold(n_splits = 5, shuffle = True, random_state = 30)

scores = cross_val_score(knn, X, y, cv = kf, scoring = 'max_error')
scores.mean()
scores = cross_val_score(knn, X, y, cv = kf, scoring = 'explained_variance')
scores.mean()
scores = cross_val_score(knn, X, y, cv = kf, scoring = 'r2')
scores.mean()
knn_params = {'n_neighbors': np.arange(1, 51)}
knn_grid = GridSearchCV(knn, 
                        knn_params, 
                        scoring='explained_variance',
                        cv = kf)
knn_grid.fit(X_train, y_train)
print("Наилучшее качество при:", knn_grid.best_params_)
print("Наилучшая оценка качества: ",knn_grid.best_score_)
results_df = pd.DataFrame(knn_grid.cv_results_)
results_df
plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])

plt.xlabel('Number of neighbors')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
param_p = np.linspace(1.0, 10.0, num = 200)
kf = KFold(n_splits = 5, shuffle = True, random_state = 11)

max_score = -1
max_score_p = 0

for p in param_p:
    knn_new = KNeighborsRegressor(n_neighbors = 44, p = p, weights='distance', metric='minkowski')

    knn_new.fit(X_train, y_train)
    y_pred_new = knn_new.predict(X_valid)

    scores = cross_val_score(knn_new, X, y, cv = kf, scoring = 'neg_mean_squared_error')
    new_score = scores.mean()
    print("P:", p, "; Score:", new_score)
    if new_score > max_score:
        max_score = new_score
        max_score_p = p
print("При p равном", max_score_p, "качество модели на кросс-валидации оказалось оптимальным --", max_score)
rnr = RadiusNeighborsRegressor(radius = 1.5)
rnr.fit(X_train, y_train)
y_pred = rnr.predict(X_valid)
y_pred
rnr = RadiusNeighborsRegressor(radius = 20)
rnr.fit(X_train, y_train)
y_pred = rnr.predict(X_valid)
y_pred
rnr.score(X_valid, y_valid)
mean_squared_error(y_valid, y_pred)