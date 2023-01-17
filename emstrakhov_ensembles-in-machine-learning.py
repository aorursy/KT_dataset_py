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
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, precision_score, recall_score, f1_score

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt
def print_results(y_true, y_pred):

    print(confusion_matrix(y_true, y_pred))

    print('F1-score:', f1_score(y_true, y_pred))
def plot_validation_curve(model_grid, param_name, params=None):

    # Рисуем валидационную кривую

    # По оси х --- значения гиперпараметров (param_***)

    # По оси y --- значения метрики (mean_test_score)



    results_df = pd.DataFrame(model_grid.cv_results_)

    

    if params == None:

        plt.plot(results_df['param_'+param_name], results_df['mean_test_score'])

    else:

        plt.plot(params, results_df['mean_test_score'])



    # Подписываем оси и график

    plt.xlabel(param_name)

    plt.ylabel('Test F1 score')

    plt.title('Validation curve')

    plt.show()
df = pd.read_csv('/kaggle/input/depression/b_depressed.csv')

df.head()
# Удалим пропуски

df_1 = df.dropna()



# Дропнем ненужные столбцы

df_2 = df_1.drop(['Survey_id', 'depressed'], axis=1)



# Переведём признаки "Номер виллы" и "Уровень образования" в бинарные 

# * мы не уверены на 100 %, что уровень образования ранговый, поэтому считаем его категориальным

df_3 = pd.get_dummies(df_2, columns=['Ville_id', 'education_level'])

df_3.head()
# Разделение на train и valid

X = df_3

y = df_1['depressed']



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25,

                                                      random_state=19)
# Масштабирование

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_valid = scaler.transform(X_valid)
ab = AdaBoostClassifier(random_state=19)

ab.fit(X_train, y_train)

y_pred = ab.predict(X_valid)

print_results(y_valid, y_pred)
# Тюнинг параметров

ab_n_estimators = {'n_estimators': np.arange(10, 100, 10)}

ab_grid = GridSearchCV(ab, ab_n_estimators, cv=5, scoring='f1', n_jobs=-1)

ab_grid.fit(X_train, y_train)



print(ab_grid.best_score_)

print(ab_grid.best_params_)
plot_validation_curve(ab_grid, 'n_estimators')
ab_n_estimators = {'n_estimators': np.arange(100, 201, 20)}

ab_grid = GridSearchCV(ab, ab_n_estimators, cv=5, scoring='f1', n_jobs=-1)

ab_grid.fit(X_train, y_train)



print(ab_grid.best_score_)

print(ab_grid.best_params_)
plot_validation_curve(ab_grid, 'n_estimators')
ab_best = ab_grid.best_estimator_

y_pred = ab_best.predict(X_valid)

print_results(y_valid, y_pred)
ab_l_rate = {'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.75, 1, 2, 10]}

ab_grid = GridSearchCV(ab, ab_l_rate, cv=5, scoring='f1', n_jobs=-1)

ab_grid.fit(X_train, y_train)



print(ab_grid.best_score_)

print(ab_grid.best_params_)
plot_validation_curve(ab_grid, 'learning_rate')
ab_best = ab_grid.best_estimator_

y_pred = ab_best.predict(X_valid)

print_results(y_valid, y_pred)
ab_params = {'n_estimators': np.arange(20, 201, 20), 

             'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.75, 1, 2, 5, 10]}

ab_grid = GridSearchCV(ab, ab_params, cv=5, scoring='f1', n_jobs=-1)

ab_grid.fit(X_train, y_train)



print(ab_grid.best_score_)

print(ab_grid.best_params_)
ab_best = ab_grid.best_estimator_

y_pred = ab_best.predict(X_valid)

print_results(y_valid, y_pred)
gb = GradientBoostingClassifier(random_state=19)

gb.fit(X_train, y_train)

y_pred = gb.predict(X_valid)

print_results(y_valid, y_pred)
gb_n_estimators = {'n_estimators': np.arange(20, 201, 20)}

gb_grid = GridSearchCV(gb, gb_n_estimators, cv=5, scoring='f1', n_jobs=-1)

gb_grid.fit(X_train, y_train)



print(gb_grid.best_score_)

print(gb_grid.best_params_)
gb_best = gb_grid.best_estimator_

y_pred = gb_best.predict(X_valid)

print_results(y_valid, y_pred)
gb_l_rate = {'learning_rate': [0.05, 0.1, 0.5, 0.75, 1, 2, 5, 10]}

gb_grid = GridSearchCV(gb, gb_l_rate, cv=5, scoring='f1', n_jobs=-1)

gb_grid.fit(X_train, y_train)



print(gb_grid.best_score_)

print(gb_grid.best_params_)
gb_best = gb_grid.best_estimator_

y_pred = gb_best.predict(X_valid)

print_results(y_valid, y_pred)
# gb_params = {'n_estimators': np.arange(20, 201, 20), 

#              'learning_rate': [0.1, 0.5, 0.75, 1, 2, 5, 10]}

# gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='f1', n_jobs=-1)

# gb_grid.fit(X_train, y_train)



# print(gb_grid.best_score_)

# print(gb_grid.best_params_)
# gb_best = gb_grid.best_estimator_

# y_pred = gb_best.predict(X_valid)

# print_results(y_valid, y_pred)
gb_max_depth = {'max_depth': np.arange(1, 11)}

gb_grid = GridSearchCV(gb, gb_max_depth, cv=5, scoring='f1', n_jobs=-1)

gb_grid.fit(X_train, y_train)



print(gb_grid.best_score_)

print(gb_grid.best_params_)
plot_validation_curve(gb_grid, 'max_depth')
gb_best = gb_grid.best_estimator_

y_pred = gb_best.predict(X_valid)

print_results(y_valid, y_pred)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_valid, y_pred)
# gb_params = {'n_estimators': np.arange(20, 221, 50), 

#              'learning_rate': [0.1, 0.5, 1, 5, 10],

#              'max_depth': np.arange(1, 11, 2)}

# gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='roc_auc', n_jobs=-1)

# gb_grid.fit(X_train, y_train)



# print(gb_grid.best_score_)

# print(gb_grid.best_params_)
# gb_best = gb_grid.best_estimator_

# y_pred = gb_best.predict(X_valid)

# print_results(y_valid, y_pred)
import xgboost as xgb

xgbc = xgb.XGBClassifier()

xgbc.fit(X_train, y_train)

y_pred = xgbc.predict(X_valid)

print_results(y_valid, y_pred)
xgb_params = {'n_estimators': [20, 50, 100, 200],

             'max_depth': [2, 4, 6]}

xgb_grid = GridSearchCV(xgbc, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)

xgb_grid.fit(X_train, y_train)



print(xgb_grid.best_score_)

print(xgb_grid.best_params_)
xgb_best = xgb_grid.best_estimator_

y_pred = xgb_best.predict(X_valid)

print_results(y_valid, y_pred)