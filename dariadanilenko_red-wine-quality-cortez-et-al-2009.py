import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head(20).T
df.info()
df_target = df['quality']

df_target
from scipy.stats import normaltest

sns.kdeplot(df_target)
data, p = normaltest(df_target)

print("p-value = ", p)
df['quality_log'] = np.log(df['quality'])

df_target_log = df['quality_log']

print(df_target)

print(df_target_log)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = df.drop('quality', axis = 1)

df_scaled = df.drop('quality_log', axis = 1)

df_scaled_fin = scaler.fit_transform(df_scaled)

df_scaled_fin
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_scaled_fin, df_target_log, test_size=0.25, random_state=412)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=100)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)
knn.score(X_valid, y_valid)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=412)

knn = KNeighborsRegressor(n_neighbors=100)

scores = cross_val_score(knn, df_scaled_fin, df_target_log, cv=kf, scoring='neg_mean_squared_error')

scores.mean()
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn, knn_params, scoring='neg_mean_squared_error', cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
results_df = pd.DataFrame(knn_grid.cv_results_)

results_df.T
import matplotlib.pyplot as plt

plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])



plt.xlabel('n_neighbors')

plt.ylabel('Test error')

plt.title('Validation curve')

plt.show()
p_params = {"p": np.linspace(1,10,200)}

knn = KNeighborsRegressor(n_neighbors = 3, weights = "distance", n_jobs = -1) #метрика Минковского идет по умолчанию

knn_cv = GridSearchCV(knn, p_params, cv = kf, scoring="neg_mean_squared_error")

knn_cv.fit(df_scaled_fin, df_target_log)
knn_cv.best_estimator_
knn_cv.best_score_
knn_cv.best_params_
knn_cv_results = pd.DataFrame(knn_cv.cv_results_)

knn_cv_results.T
plt.plot(knn_cv_results["param_p"],knn_cv_results["mean_test_score"])

plt.xlabel('n_neighbors')

plt.ylabel('Test error')

plt.title('Validation curve')

plt.show()
from sklearn.neighbors import RadiusNeighborsRegressor

rnr = RadiusNeighborsRegressor(radius = 7)

rnr.fit(X_train, y_train)

y_pred = rnr.predict(X_valid)

y_pred
rnr.score(X_valid, y_valid)
mean_squared_error(y_valid, y_pred)