import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

sns.set(style="ticks")
from sklearn.model_selection import train_test_split

from sklearn import neighbors 

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedKFold

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score 

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import learning_curve, validation_curve
from sklearn import linear_model

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import BaggingRegressor
data = pd.read_csv('../input/weatherHistory.csv', sep=",")
data
#датасет не имеет пропущенных значений категориальных признаков

data.info() 
data.describe()
sns.pairplot(data[["Precip Type","Temperature (C)","Apparent Temperature (C)","Humidity"]],

             hue="Precip Type",

             palette="YlGnBu");
fig, ax = plt.subplots(figsize=(10,10)) 

sns.distplot(data['Temperature (C)'])
fig, ax = plt.subplots(figsize=(40,20)) 

sns.scatterplot(ax=ax, x=data.Summary[data['Temperature (C)']>0], y='Temperature (C)', data=data)
sns.set(rc={'figure.figsize':(30,10)}, font_scale=1.5, style='whitegrid')

ax = sns.boxplot(x="Humidity",y="Temperature (C)",data=data)

labels = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right')
sns.heatmap(data.corr(), annot=True, fmt='.3f')
data.corr()
# преобразование поля Formatted Date

datetime = pd.to_datetime(data["Formatted Date"])

datetime = datetime.apply(lambda x: x+pd.Timedelta(hours=2))

data["Month"] = datetime.apply(lambda x: x.month)

data["Day"] = datetime.apply(lambda x: x.day)

data["WoY"] = datetime.apply(lambda x: x.week)

data["Hour"] = datetime.apply(lambda x: x.hour)
data = data.drop('Formatted Date', axis=1)

data = data.drop('Loud Cover', axis=1)

data = data.drop('Daily Summary', axis=1)

data = data.dropna(axis=0, how='any')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Precip Type'] = le.fit_transform(data['Precip Type'])

data['Summary'] = le.fit_transform(data['Summary'])
data.head()
sns.heatmap(data.corr())
temp = data['Temperature (C)']

data = data.drop('Temperature (C)', axis=1)

data = data.drop('Apparent Temperature (C)', axis=1)

print(data.head())

print(temp.head())
x_train, x_test, y_train, y_test = train_test_split(

    data, temp, test_size=0.3, random_state=1)
knn_start = neighbors.KNeighborsRegressor(n_neighbors=5)

knn_start.fit(x_train, y_train)

target1_0 = knn_start.predict(x_train)

target1_1 = knn_start.predict(x_test)
print('mae for x_train: ',mean_absolute_error(y_train, target1_0))

print('mae for x_test: ',mean_absolute_error(y_test, target1_1))

print('mse: ', mean_squared_error(y_test, target1_1))

print('explained variance score: ', explained_variance_score(y_test, target1_1))

print('R^2 score: ', r2_score(y_test, target1_1))
br_start = linear_model.BayesianRidge()

br_start.fit(x_train, y_train)

target_br_start = br_start.predict(x_test)
print('mae for x_test: ',mean_absolute_error(y_test, target_br_start))

print('mse: ', mean_squared_error(y_test, target_br_start))

print('explained variance score: ', explained_variance_score(y_test, target_br_start))

print('R^2 score: ', r2_score(y_test, target_br_start))
rf_start = RandomForestRegressor()

rf_start.fit(x_train, y_train)

target_rf_start = rf_start.predict(x_test)
print('mae for x_test: ',mean_absolute_error(y_test, target_rf_start))

print('mse: ', mean_squared_error(y_test, target_rf_start))

print('explained variance score: ', explained_variance_score(y_test, target_rf_start))

print('R^2 score: ', r2_score(y_test, target_rf_start))
gb_start = GradientBoostingRegressor()

gb_start.fit(x_train, y_train)

target_gb_start = gb_start.predict(x_test)
print('mae for x_test: ',mean_absolute_error(y_test, target_gb_start))

print('mse: ', mean_squared_error(y_test, target_gb_start))

print('explained variance score: ', explained_variance_score(y_test, target_gb_start))

print('R^2 score: ', r2_score(y_test, target_gb_start))
br_start = BaggingRegressor()

br_start.fit(x_train, y_train)

target_br_start = br_start.predict(x_test)
print('mae for x_test: ',mean_absolute_error(y_test, target_br_start))

print('mse: ', mean_squared_error(y_test, target_br_start))

print('explained variance score: ', explained_variance_score(y_test, target_br_start))

print('R^2 score: ', r2_score(y_test, target_br_start))
tuned_parameters = {'n_neighbors': range(1,30,5)}

clf_gs = GridSearchCV(neighbors.KNeighborsRegressor(), tuned_parameters, cv=5)

clf_gs.fit(x_train, y_train)
clf_gs.best_estimator_
clf_gs.best_score_
cls_best_knn = clf_gs.best_estimator_.fit(x_train, y_train)

target1_1_knn = cls_best_knn.predict(x_test)

print('mae for x_test: ',mean_absolute_error(y_test, target1_1_knn))

print('mse: ', mean_squared_error(y_test, target1_1_knn))

print('explained variance score: ', explained_variance_score(y_test, target1_1_knn))

print('R^2 score: ', r2_score(y_test, target1_1_knn))
tuned_parameters = {'n_iter': range(100,500,50)}

clf_brd = GridSearchCV(linear_model.BayesianRidge(), tuned_parameters, cv=5)

clf_brd.fit(x_train, y_train)
clf_brd.best_estimator_
clf_brd.best_score_
cls_best_br = clf_brd.best_estimator_.fit(x_train, y_train)

target1_1_br = cls_best_br.predict(x_test)

print('mae for x_test: ',mean_absolute_error(y_test, target1_1_br))

print('mse: ', mean_squared_error(y_test, target1_1_br))

print('explained variance score: ', explained_variance_score(y_test, target1_1_br))

print('R^2 score: ', r2_score(y_test, target1_1_br))
parameters_random_forest = {'n_estimators':[1, 3, 5, 7, 10], 

                            'max_depth':[1, 3, 5, 7, 10],

                            'random_state':[0, 2, 4, 6, 8, 10],

                           'max_features':['auto', 'sqrt']}

best_random_forest = GridSearchCV(RandomForestRegressor(), parameters_random_forest, cv=3)

best_random_forest.fit(x_train, y_train)
best_random_forest.best_params_
best_random_forest.best_score_
cls_best_rf = best_random_forest.best_estimator_.fit(x_train, y_train)

target1_1_rf = cls_best_rf.predict(x_test)

print('mae for x_test: ',mean_absolute_error(y_test, target1_1_rf))

print('mse: ', mean_squared_error(y_test, target1_1_rf))

print('explained variance score: ', explained_variance_score(y_test, target1_1_rf))

print('R^2 score: ', r2_score(y_test, target1_1_rf))
parameters_gradient_boosting = {'n_estimators':[1, 3, 5, 7, 10], 

                            'max_depth':[1, 3, 5, 7, 10],

                            'learning_rate':[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025]}

best_gradient_boosting = GridSearchCV(GradientBoostingRegressor(), parameters_gradient_boosting, cv=3)

best_gradient_boosting.fit(x_train, y_train)
best_gradient_boosting.best_params_
best_gradient_boosting.best_score_
cls_best_gb = best_gradient_boosting.best_estimator_.fit(x_train, y_train)

target1_1_gb = cls_best_gb.predict(x_test)

print('mae for x_test: ',mean_absolute_error(y_test, target1_1_gb))

print('mse: ', mean_squared_error(y_test, target1_1_gb))

print('explained variance score: ', explained_variance_score(y_test, target1_1_gb))

print('R^2 score: ', r2_score(y_test, target1_1_gb))
tuned_parameters = {'max_samples' : [0.05, 0.1, 0.2, 0.5]}

clf_bg = GridSearchCV(BaggingRegressor(), tuned_parameters, cv=5)

clf_bg.fit(x_train, y_train)
clf_bg.best_params_
clf_bg.best_score_
cls_best_bg = clf_bg.best_estimator_.fit(x_train, y_train)

target1_1_bg = cls_best_bg.predict(x_test)

print('mae for x_test: ',mean_absolute_error(y_test, target1_1_bg))

print('mse: ', mean_squared_error(y_test, target1_1_bg))

print('explained variance score: ', explained_variance_score(y_test, target1_1_bg))

print('R^2 score: ', r2_score(y_test, target1_1_bg))
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
plot_learning_curve(clf_gs.best_estimator_, 'n_neighbors=20',

                   x_test, y_test)
plot_learning_curve(clf_brd.best_estimator_, "Bayessian Ridge", x_test, y_test)
plot_learning_curve(best_random_forest.best_estimator_, "Random Forest", x_test, y_test)
plot_learning_curve(best_gradient_boosting.best_estimator_, "Gradient Boosting", x_test, y_test)
plot_learning_curve(clf_bg.best_estimator_, "Bagging", x_test, y_test)


def plot_validation_curve(estimator, title, X, y, 

                          param_name, param_range, cv, 

                          scoring="accuracy"):

                                                   

    train_scores, test_scores = validation_curve(

        estimator, X, y, param_name=param_name, param_range=param_range,

        cv=cv, scoring=scoring, n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.title(title)

    plt.xlabel(param_name)

    plt.ylabel("Score")

    plt.ylim(-31.1, 0.0)

    lw = 2

    plt.plot(param_range, train_scores_mean, label="Training score",

                 color="darkorange", lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.2,

                     color="darkorange", lw=lw)

    plt.plot(param_range, test_scores_mean, label="Cross-validation score",

                 color="navy", lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.2,

                     color="navy", lw=lw)

    plt.legend(loc="best")

    return plt
plot_validation_curve(clf_gs.best_estimator_, 'knn', x_train, y_train, param_name='n_neighbors', 

                      param_range=[1,3,5,7,10], cv=3,  scoring="neg_mean_squared_error")
plot_validation_curve(clf_brd.best_estimator_, 'Bayessian Ridge', x_train, y_train, param_name='n_iter', 

                      param_range=range(100,500,50), cv=3,  scoring="neg_mean_squared_error")
plot_validation_curve(best_random_forest.best_estimator_, 'Random Forest', x_train, y_train, param_name='n_estimators', 

                      param_range=[1,3,5,7,10], cv=3,  scoring="neg_mean_squared_error")
plot_validation_curve(best_gradient_boosting.best_estimator_, 'Gradient Boosting', x_train, y_train, param_name='n_estimators', 

                      param_range=[1,3,5,7,10], cv=3,  scoring="neg_mean_squared_error")
plot_validation_curve(clf_bg.best_estimator_, 'Bagging', x_train, y_train, param_name='max_samples', 

                      param_range=[0.05, 0.1, 0.2, 0.5], cv=3,  scoring="neg_mean_squared_error")