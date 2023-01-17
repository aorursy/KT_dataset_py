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

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt



np.random.seed(1)
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



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)
# Масштабирование

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_valid = scaler.transform(X_valid)
mlp = MLPClassifier(solver='lbfgs')

mlp.fit(X_train, y_train)



y_pred = mlp.predict(X_valid)

print_results(y_valid, y_pred) # log_reg: ~0.2
X_sc = scaler.fit_transform(X)

X_train1, X_valid1, y_train1, y_valid1 = train_test_split(X_sc, y, test_size=0.25, random_state=1)
mlp = MLPClassifier(solver='lbfgs')

mlp.fit(X_train1, y_train1)



y_pred1 = mlp.predict(X_valid1)

print_results(y_valid1, y_pred1) # log_reg: ~0.2
mlp_2 = MLPClassifier(hidden_layer_sizes=(200,), solver='lbfgs', max_iter=400, alpha=0.1)

mlp_2.fit(X_train, y_train)



y_pred = mlp.predict(X_valid)

print_results(y_pred, y_valid) # log_reg: ~0.2
mlp_2.n_iter_
ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X_train, y_train)
mlp_3 = MLPClassifier(hidden_layer_sizes=(200,), solver='lbfgs', max_iter=400, alpha=0.1)

mlp_3.fit(X_ros, y_ros)



y_pred = mlp_3.predict(X_valid)

print_results(y_pred, y_valid) # log_reg: ~0.2
mlp_4 = MLPClassifier(hidden_layer_sizes=(100, 50, 20), solver='lbfgs', alpha=0.001)

mlp_4.fit(X_ros, y_ros)



y_pred = mlp_4.predict(X_valid)

print(confusion_matrix(y_valid, y_pred))

print('F1-score:', f1_score(y_valid, y_pred)) # log_reg: ~0.2
scaler = StandardScaler()

mlp = MLPClassifier(solver='lbfgs')

model = Pipeline([('scaler', scaler), ('mlp', mlp)])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print('F1-score:', f1_score(y_test, y_pred))
param_grid = {'mlp__alpha': np.logspace(-4, 4, 10)}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', 

                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров

model_grid.fit(X_train, y_train)
plot_validation_curve(model_grid, 'mlp__alpha')
print('Best (hyper)parameters:', model_grid.best_params_)

print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print('F1-score:', f1_score(y_test, y_pred))
X_ros, y_ros = ros.fit_sample(X_train, y_train)

model_grid.fit(X_ros, y_ros)
print('Best (hyper)parameters:', model_grid.best_params_)

print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print('F1-score:', f1_score(y_test, y_pred))
param_grid = {'mlp__activation': ['logistic', 'tanh', 'relu']}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', 

                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров

model_grid.fit(X_train, y_train)
plot_validation_curve(model_grid, 'mlp__activation')
print('Best (hyper)parameters:', model_grid.best_params_)

print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_test)

print_results(y_test, y_pred)
param_grid = {'mlp__hidden_layer_sizes': [(i, ) for i in range(20, 500, 20)]}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', 

                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров

model_grid.fit(X_train, y_train)
plot_validation_curve(model_grid, 'mlp__hidden_layer_sizes', 

                      [i for i in range(20, 500, 20)])
print('Best (hyper)parameters:', model_grid.best_params_)

print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_test)

print_results(y_test, y_pred)
param_grid = {'mlp__warm_start': [True, False]}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', 

                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров

model_grid.fit(X_train, y_train)
print('Best (hyper)parameters:', model_grid.best_params_)

print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_test)

print_results(y_test, y_pred)
hidden = [(100,), (100, 50), (100, 50, 20), (50, 50), (50, 50, 50), (50, 30, 30, 20)]

param_grid = {'mlp__hidden_layer_sizes': hidden}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', 

                          n_jobs=-1) # n_jobs=-1 задействует больше процессоров

model_grid.fit(X_train, y_train)
plot_validation_curve(model_grid, 'mlp__hidden_layer_sizes', 

                      [str(x) for x in hidden])
print('Best (hyper)parameters:', model_grid.best_params_)

print('Best score:', model_grid.best_score_)
y_pred = model_grid.best_estimator_.predict(X_test)

print_results(y_test, y_pred)