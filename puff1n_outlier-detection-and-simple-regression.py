

import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input/brasilian-houses-to-rent/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt

import seaborn as sns

plt.set_cmap('Dark2')
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

df.info()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].scatter(df['area'], df['total (R$)'])

ax[1].scatter(np.log(df['area']), np.log(df['total (R$)']))



ax[0].set_title('Base area and total $')

ax[1].set_title('Log scale')



ax[0].set_xlabel('Area');

ax[0].set_ylabel('Total $');



ax[1].set_xlabel('Area');

ax[1].set_ylabel('Total $');
# Prepare data for illustration

x = np.random.normal(0, 1, 950)

x_out = np.random.randint(6, 10, 50)

x_final = np.append(x_out, x)

x_final_std = np.std(x_final); x_final_mean = np.mean(x_final)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))

ax1.axvspan(x_final_mean + 3*x_final_std, 

            x_final_mean - 3*x_final_std, alpha=.2, color='r')

ax1.axvspan(x_final_mean + 2*x_final_std, 

            x_final_mean - 2*x_final_std, alpha=.2, color='g')

ax1.axvspan(x_final_mean + 1*x_final_std, 

            x_final_mean - 1*x_final_std, alpha=.2, color='blue')

ax1.axvspan(x_final_mean-.05, 

            x_final_mean+.05, color='navy')

ax1.text(8, .25, 'Outliers')

ax1.text(4.5, .25, '3 stds')

ax1.text(2.5, .25, '2 stds')

ax1.text(1, .25, '1 std')

ax1.set_title('Standart deviation approach')

sns.distplot(x_final, ax=ax1, color='black')



ax2.axvspan(max(x_final[x_final < np.quantile(x_final, .05)]), 

            min(x_final[x_final > np.quantile(x_final, 1-.05)]), 

            alpha=.2, color='r')

ax2.axvspan(max(x_final[x_final < np.quantile(x_final, .15)]), 

            min(x_final[x_final > np.quantile(x_final, 1-.15)]), 

            alpha=.2, color='g')

ax2.axvspan(max(x_final[x_final < np.quantile(x_final, .25)]), 

            min(x_final[x_final > np.quantile(x_final, 1-.25)]), 

            alpha=.2, color='b')



ax2.text(8, .25, 'Outliers')

ax2.text(4, .25, '95% quantile')

ax2.text(1, .35, '85% quantile',rotation=90)

ax2.text(.3, .25, '75% quantile',rotation=90)

ax2.set_title('Quantile approach')

sns.distplot(x_final, ax=ax2, color='black')
def quantile_outlier(x, t=.25):

    x = np.array(x)

    low_thr = np.quantile(x, t)

    upp_thr = np.quantile(x, 1-t)

    return np.array((low_thr < x) & (x < upp_thr), dtype=int)



def std_outlier(x, t):

    mu = x.mean()

    low_thr = mu - t * x.std()

    upp_thr = mu + t * x.std()

    return np.array((low_thr < x) & (x < upp_thr), dtype=int)
iso = IsolationForest(n_estimators=300, contamination=.01, bootstrap=True)

lof = LocalOutlierFactor(n_neighbors=250, algorithm='brute', contamination=.005)



# Make predictions

y_pred_lof = lof.fit_predict(df[['area', 'total (R$)']])

y_pred_iso = iso.fit_predict(df[['area', 'total (R$)']])

y_pred_qua = quantile_outlier(df['total (R$)'], .01) & quantile_outlier(df['area'], .01)

y_pred_std = std_outlier(df['total (R$)'], 3) & std_outlier(df['area'], 1)
fig, ax = plt.subplots(2, 2, figsize=(20, 10))



ax[0, 0].scatter(np.log(df['area']), np.log(df['total (R$)']), 

                 c=y_pred_iso, marker='.')

ax[0, 0].set_title('IsolationForest')

ax[0, 1].scatter(np.log(df['area']), np.log(df['total (R$)']), 

                 c=y_pred_lof, marker='.')

ax[0, 1].set_title('LocalOutlierFactor')

ax[1, 0].scatter(np.log(df['area']), np.log(df['total (R$)']), 

                 c=y_pred_qua, marker='.')

ax[1, 0].set_title('Quantile based')

ax[1, 1].scatter(np.log(df['area']), np.log(df['total (R$)']), 

                 c=y_pred_std, marker='.')

ax[1, 1].set_title('Std based')

plt.xlabel('area');

plt.ylabel('total$');

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler
df['total (R$)'] = np.log(df['total (R$)'])
df_tr = pd.get_dummies(df, columns=['city', 'animal', 'furniture'], prefix='d')

df_tr['floor'].replace('0', 0, inplace=True)

df_tr.replace('-', 0, inplace=True)
X = df_tr.drop('total (R$)',axis=1)

y = df['total (R$)']
# Remove ouliers and make filtered datasets

lof = LocalOutlierFactor()

mask = np.array(lof.fit_predict(X)+1, dtype='bool')

X_filtered = X[mask]

y_filtered = y[mask]
lr = LinearRegression()

rid = Ridge()

las = Lasso()

ela = ElasticNet()

rnf = RandomForestRegressor()





params_lr = {

    'normalize' : [True, False]

}

params_rid = {

    'alpha' : [.0, .01, .05, .5, 1], 

    'normalize' : [True, False]

}

params_las = {

    'alpha' : [.0, .01, .05, .5, 1], 

    'normalize' : [True, False]

}

params_ela = {

    'alpha' : [.0, .01, .05, .5, 1], 

    'l1_ratio' : [.0, .01, .05, .5, 1, 2],

    'normalize' : [True, False]

}

params_rnf = {

    #'n_estimators' : np.arange(1, 150, 10), #for time reasons

    #'max_depth' : [1, 5, 10, 20, 35, 50]

    'max_depth': [20, 50], 'n_estimators': [31, 141]

}

def get_grid_results(estimator, params, X, y, ncv=6):

    gs = GridSearchCV(estimator, params, cv=ncv, n_jobs=-1, 

                      return_train_score=True, 

                      scoring='neg_mean_squared_error')

    gs.fit(X, y)

    print(gs.best_params_)

    return gs
clf_names = ['lr', 'rid', 'las', 'ela', 'rnf']

mean_train_scores = []

mean_test_scores = []

best_clfs = []



mean_train_scores_filtered = []

mean_test_scores_filtered = []

best_clfs_filtered = []



for c, p in zip([lr, rid, las, ela, rnf], 

                [params_lr, params_rid, params_las, params_ela, params_rnf]):

    clf = get_grid_results(c, p, X, y)

    mean_train_scores.append(-np.mean(clf.cv_results_['mean_train_score']))

    mean_test_scores.append(-np.mean(clf.cv_results_['mean_test_score']))

    best_clfs.append(clf.best_estimator_)

    

    clf_filtered = get_grid_results(c, p, X_filtered, y_filtered)

    mean_train_scores_filtered.append(-np.mean(clf_filtered.cv_results_['mean_train_score']))

    mean_test_scores_filtered.append(-np.mean(clf_filtered.cv_results_['mean_test_score']))

    best_clfs_filtered.append(clf_filtered.best_estimator_)
fig, (ax1) = plt.subplots(1, 1, figsize=(14, 5))

ax1.bar(np.arange(len(mean_train_scores))-.3, 

        mean_train_scores, width=.2, 

        label='Mean train full', color='b')

ax1.bar(np.arange(len(mean_test_scores))-.1,

        mean_test_scores, width=.2, 

        label='Mean test full', color='b',alpha=.5)

ax1.bar(np.arange(len(mean_train_scores_filtered))+.1, 

        mean_train_scores_filtered, width=.2, 

        label='Mean train filtered', color='r')

ax1.bar(np.arange(len(mean_test_scores_filtered))+.3, 

        mean_test_scores_filtered, 

        width=.2, label='Mean test filtered', color='r', alpha=.5)



ax1.set_title('Mean train and test scores on full dataset and filtered datasets')

ax1.set_xticks(np.arange(len(clf_names)));

ax1.set_xticklabels(clf_names, fontsize=16)

ax1.set_yticks(np.arange(0, .6, .1));

ax1.set_yticklabels([f'{i:.1f}' for i in np.arange(0, .6, .10, )], fontsize=16)

ax1.grid();

ax1.legend();