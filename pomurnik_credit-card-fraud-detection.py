import numpy as np 

import pandas as pd 

import time

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.preprocessing import RobustScaler

from lightgbm import LGBMClassifier

from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.cluster import KMeans

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")

from warnings import simplefilter

from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

simplefilter("ignore", category=UserWarning)

from pandas.core.common import SettingWithCopyWarning

simplefilter(action="ignore", category=SettingWithCopyWarning)
X_all_data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

X_all_data.info()
X_all_data.isnull().sum()
y_all_data = X_all_data.Class

X_all_data.drop(['Class'], axis=1, inplace=True)

X_all_data.head()
y_all_data.value_counts()
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X_all_data, y_all_data, train_size=0.85, test_size=0.15, random_state=1)
fig = plt.figure(figsize=(18,22))

for index in range(len(X_train_valid.columns)):

    plt.subplot(10,3,index+1)

    sns.distplot(X_train_valid.iloc[:,index].dropna(), norm_hist=False)

fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(18,22))

for index in range(len(X_train_valid.columns)):

    plt.subplot(10,3,index+1)

    sns.boxplot(y=X_train_valid.iloc[:,index], data=X_train_valid.dropna())

fig.tight_layout(pad=1.0)
plt.figure(figsize=(18,16))

correlation = X_train_valid.corr()

sns.heatmap(correlation, mask = correlation <0.1, linewidth=0.5, cmap='Blues', annot=True)
sm = SMOTE(sampling_strategy = 0.1, n_jobs=-1, random_state = 1)
def evaluate_model(model, name):

    model_performances = pd.DataFrame({

        'Model' : [name],

        'Mean F1 score(val)' : round(model.cv_results_['mean_test_f1'][model.best_index_], 3),

        'Mean Precision(val)': round(model.cv_results_['mean_test_precision'][model.best_index_], 3),

        'Mean Recall(val)': round(model.cv_results_['mean_test_recall'][model.best_index_], 3),

        'Fit time(val)': round(model.cv_results_['mean_fit_time'][model.best_index_], 3)

    })

    model_performances.set_index('Model', inplace=True, drop=True)

    return model_performances
kf = StratifiedKFold(5, shuffle=True, random_state=1)
start = time.time()



lgbm = Pipeline([

        ('sampling', sm),

        ('scaling', RobustScaler()),

        ('model', LGBMClassifier())

    ])



param_lst_lgbm = {

    'model__max_depth' : [2, 3, 5, 7, 8],

    'model__num_leaves' : [3, 5, 20, 80, 180],

    'model__learning_rate' : [0.001, 0.01, 0.1, 0.2],

    'model__n_estimators' : [100, 300, 500, 1000, 1500, 2000],

    'model__reg_alpha' : [0.001, 0.01, 1, 10, 100],

    'model__reg_lambda' : [0.001, 0.01, 1, 10, 100],

    'model__colsample_bytree' : [0.5, 0.7, 0.8],

    'model__min_child_samples' : [5, 10, 20, 25],

}



lgbm_cv = RandomizedSearchCV(estimator = lgbm,

                              param_distributions = param_lst_lgbm,

                              n_iter = 30,

                              scoring = ['f1', 'precision', 'recall'],

                              refit='f1',

                              cv = kf,

                              n_jobs = -1)

       

lgbm_search = lgbm_cv.fit(X_train_valid, y_train_valid)



best_param_lgbm = lgbm_search.best_params_

end = time.time()

time_lgbm = round(end-start, 0)

print('Dobrane parametry dla modelu LGBM: ', best_param_lgbm)

print('Czas dopierania parametrów: ', time_lgbm, ' [sek]')
cv_lgbm = evaluate_model(lgbm_search,  'LGBM')

print(cv_lgbm)
def evaluate_model_test(model, X, y, name):

    start = time.time()

    pred = model.predict(X) 

    end = time.time()

    f1_sc = f1_score(y, pred)

    precision_sc = precision_score(y, pred)

    recall_sc = recall_score(y, pred)

    model_performances_test = pd.DataFrame({

        'Model' : [name],

        'F1 score(test)' : round(f1_sc, 3),

        'Precision(test)': round(precision_sc, 3),

        'Recall(test)': round(recall_sc, 3),

        'Predict time' : round(end-start, 4)

    })

    model_performances_test.set_index('Model', inplace=True, drop=True)

    return model_performances_test
test_lgbm = evaluate_model_test(lgbm_search, X_test, y_test, 'LGBM')

results_all = pd.concat([cv_lgbm, test_lgbm ], axis=1)

results_all.head()
feature_imp = pd.DataFrame(sorted(zip(lgbm_search.best_estimator_.named_steps['model'].feature_importances_,X_test.columns)), columns=['Value','Feature'])



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()