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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

def submit_csv(predict):
    result = pd.DataFrame(columns=['id','label'])
    result['id'] = np.arange(1,len(predict)+1)
    result['label'] = predict
    result.to_csv('answer.csv',index=False, sep=',', header=True)
    
def get_result_from_grid_cv(search, gr_cv):
    f1_macro = search['mean_test_f1_macro'][np.argmin(search['rank_test_f1_macro'])]
    precision_macro = search['mean_test_precision_macro'][np.argmin(search['rank_test_precision_macro'])]
    recall_macro = search['mean_test_recall_macro'][np.argmin(search['rank_test_recall_macro'])]
    print(gr_cv.best_params_,'\n', 'best_accuracy: ', gr_cv.best_score_)
    print('f1_macro_best:', f1_macro, '\n', 'precision_macro_best:', precision_macro,'\n', 'recall_macro_best:', recall_macro)
df = pd.read_csv('fashion-mnist_train.csv')
df_test = pd.read_csv('new_test.csv')
df_test.head()
X_valid = np.array(df_test)
X = np.array(df.loc[:, 'pixel1':])
y = np.array(df.loc[:, 'label'])
X_valid
# Стратегия кросс-валидации
# 3 разбиения для такого большого датасета хватит, иначе комп не тянет
cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
# Метрики качеств
scorer = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
# Далее поиск по сетке будет осуществляться рандомным выбором точек в сетке, 
# или, если алгоритм не долго обучается, полным перебором по сетке

# если использовать кросс-валидацию, то данные следует разбить на 2 части. Первая из
# них будет использоваться для обучения алгоритмов и оценки качества с помощью кросс-валидации, после
# чего лучший алгоритм будет проверен на адекватность на контрольной выборке. Но тут я этого делать не стану,
# так как контрольная выборка на Каггле
LogReg = LogisticRegression(n_jobs=-1)
model_log_reg = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', LogReg)])
model_log_reg.get_params().keys() # Глянем какие параметры доступны для Тюнинга
parameters = {
    'classifier__C': [0.0001, 0.001, 0.01, 10, 0.0005],
    'classifier__solver': ['saga', 'sag', 'newton-cg']
}
grid_cv_lr = GridSearchCV(model_log_reg, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_lr = RandomizedSearchCV(model_log_reg, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=20)
%%time
grid_cv_lr.fit(X, y)
result_lr_cv_search = grid_cv_lr.cv_results_
get_result_from_grid_cv(result_lr_cv_search, grid_cv_lr)
knn = KNN(n_jobs=-1)
model_knn = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', knn)])
model_knn.get_params().keys() # Глянем какие параметры доступны для Тюнинга
parameters = {
    'classifier__n_neighbors': [1, 3, 5, 7, 9, 12, 15, 35],
    'classifier__p': [1, 2],
    'classifier__algorithm' : ['ball_tree', 'kd_tree', 'brute']
}
# grid_cv_knn = GridSearchCV(model_knn, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
random_cv_knn = RandomizedSearchCV(model_knn, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
random_cv_knn.fit(X, y)
result_knn_cv_search = random_cv_knn.cv_results_
get_result_from_grid_cv(result_knn_cv_search, random_cv_knn)
GausNB = GaussianNB()
model_GausNB = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', GausNB)])
model_GausNB.get_params().keys() # Глянем какие параметры доступны для Тюнинга
parameters = {
    'classifier__var_smoothing': [0.1, 0.2, 0.5, 1]
}
grid_cv_GB = GridSearchCV(model_GausNB, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_GB = RandomizedSearchCV(model_GausNB, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
grid_cv_GB.fit(X, y)
result_GB_cv_search = grid_cv_GB.cv_results_
get_result_from_grid_cv(result_GB_cv_search, grid_cv_GB)
linear_SVC = LinearSVC(dual=True, penalty='l2')
model_linear_SVC = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', linear_SVC)])
model_linear_SVC.get_params().keys() # Глянем какие параметры доступны для Тюнинга
# когда C большой, он будет корректно классифицировать все точки данных, то есть оверфитить модель
# при C->inf будет соответствовать hard-margin svm
parameters = {
    'classifier__loss':['hinge', 'squared_hinge'],
    'classifier__C':[0.1, 0.001, 0.01, 0.0009, 0.0005]
}
grid_cv_LSVC = GridSearchCV(model_linear_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_LSVC = RandomizedSearchCV(model_linear_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
grid_cv_LSVC.fit(X, y)
result_LSVC_cv_search = grid_cv_LSVC.cv_results_
get_result_from_grid_cv(result_LSVC_cv_search, grid_cv_LSVC)
poly_SVC = SVC(random_state=42, kernel='poly', cache_size=500)
model_poly_SVC = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', poly_SVC)])
model_poly_SVC.get_params().keys() # Глянем какие параметры доступны для Тюнинга
# когда C большой, он будет корректно классифицировать все точки данных, то есть оверфитить модель
# при C->inf будет соответствовать hard-margin svm
# coef0 контролирует, насколько на модель влияют полиномы высокой степени.
# gammaо пределяет, какое влияние оказывает отдельный учебный пример. 
#                  Чем больше gamma, тем ближе другие примеры должны быть затронуты.

parameters = {
    'classifier__degree':[2,3],
    'classifier__C':[0.0005, 0.005, 0.001, 0.01],
    'classifier__gamma':[1000, 300, 100, 10],
    'classifier__coef0': [0,1,10]
}
grid_cv_PSVC = GridSearchCV(model_poly_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_PSVC = RandomizedSearchCV(model_poly_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
grid_cv_PSVC.fit(X, y)
result_PSVC_cv_search = grid_cv_PSVC.cv_results_
get_result_from_grid_cv(result_PSVC_cv_search, grid_cv_PSVC)
# SPCA, t-sne
# спецкурс по обработке изображений ВМК МГУ Конушин
lda = LDA()
model_LDA = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', lda)])
model_LDA.get_params().keys() # Глянем какие параметры доступны для Тюнинга
%%time
lda_scoring = cross_val_score(model_LDA, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy mean: {}, std: {}'.format(lda_scoring.mean(), lda_scoring.std()))
# -
# -
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

def submit_csv(predict):
    result = pd.DataFrame(columns=['id','label'])
    result['id'] = np.arange(1,len(predict)+1)
    result['label'] = predict
    result.to_csv('answer.csv',index=False, sep=',', header=True)
    
def get_result_from_grid_cv(search, gr_cv):
    f1_macro = search['mean_test_f1_macro'][np.argmin(search['rank_test_f1_macro'])]
    precision_macro = search['mean_test_precision_macro'][np.argmin(search['rank_test_precision_macro'])]
    recall_macro = search['mean_test_recall_macro'][np.argmin(search['rank_test_recall_macro'])]
    print(gr_cv.best_params_,'\n', 'best_accuracy: ', gr_cv.best_score_)
    print('f1_macro_best:', f1_macro, '\n', 'precision_macro_best:', precision_macro,'\n', 'recall_macro_best:', recall_macro)
df = pd.read_csv('fashion-mnist_train.csv')
df_test = pd.read_csv('new_test.csv')
df_test.head()
X_valid = np.array(df_test)
X = np.array(df.loc[:, 'pixel1':])
y = np.array(df.loc[:, 'label'])
X_valid
# Стратегия кросс-валидации
# 3 разбиения для такого большого датасета хватит, иначе комп не тянет
cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
# Метрики качеств
scorer = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
# Далее поиск по сетке будет осуществляться рандомным выбором точек в сетке, 
# или, если алгоритм не долго обучается, полным перебором по сетке

# если использовать кросс-валидацию, то данные следует разбить на 2 части. Первая из
# них будет использоваться для обучения алгоритмов и оценки качества с помощью кросс-валидации, после
# чего лучший алгоритм будет проверен на адекватность на контрольной выборке. Но тут я этого делать не стану,
# так как контрольная выборка на Каггле
LogReg = LogisticRegression(n_jobs=-1)
model_log_reg = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', LogReg)])
model_log_reg.get_params().keys() # Глянем какие параметры доступны для Тюнинга
parameters = {
    'classifier__C': [0.0001, 0.001, 0.01, 10, 0.0005],
    'classifier__solver': ['saga', 'sag', 'newton-cg']
}
grid_cv_lr = GridSearchCV(model_log_reg, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_lr = RandomizedSearchCV(model_log_reg, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=20)
%%time
grid_cv_lr.fit(X, y)
result_lr_cv_search = grid_cv_lr.cv_results_
get_result_from_grid_cv(result_lr_cv_search, grid_cv_lr)
knn = KNN(n_jobs=-1)
model_knn = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', knn)])
model_knn.get_params().keys() # Глянем какие параметры доступны для Тюнинга
parameters = {
    'classifier__n_neighbors': [1, 3, 5, 7, 9, 12, 15, 35],
    'classifier__p': [1, 2],
    'classifier__algorithm' : ['ball_tree', 'kd_tree', 'brute']
}
# grid_cv_knn = GridSearchCV(model_knn, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
random_cv_knn = RandomizedSearchCV(model_knn, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
random_cv_knn.fit(X, y)
result_knn_cv_search = random_cv_knn.cv_results_
get_result_from_grid_cv(result_knn_cv_search, random_cv_knn)
GausNB = GaussianNB()
model_GausNB = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', GausNB)])
model_GausNB.get_params().keys() # Глянем какие параметры доступны для Тюнинга
parameters = {
    'classifier__var_smoothing': [0.1, 0.2, 0.5, 1]
}
grid_cv_GB = GridSearchCV(model_GausNB, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_GB = RandomizedSearchCV(model_GausNB, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
grid_cv_GB.fit(X, y)
result_GB_cv_search = grid_cv_GB.cv_results_
get_result_from_grid_cv(result_GB_cv_search, grid_cv_GB)
linear_SVC = LinearSVC(dual=True, penalty='l2')
model_linear_SVC = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', linear_SVC)])
model_linear_SVC.get_params().keys() # Глянем какие параметры доступны для Тюнинга
# когда C большой, он будет корректно классифицировать все точки данных, то есть оверфитить модель
# при C->inf будет соответствовать hard-margin svm
parameters = {
    'classifier__loss':['hinge', 'squared_hinge'],
    'classifier__C':[0.1, 0.001, 0.01, 0.0009, 0.0005]
}
grid_cv_LSVC = GridSearchCV(model_linear_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_LSVC = RandomizedSearchCV(model_linear_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
grid_cv_LSVC.fit(X, y)
result_LSVC_cv_search = grid_cv_LSVC.cv_results_
get_result_from_grid_cv(result_LSVC_cv_search, grid_cv_LSVC)
poly_SVC = SVC(random_state=42, kernel='poly', cache_size=500)
model_poly_SVC = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', poly_SVC)])
model_poly_SVC.get_params().keys() # Глянем какие параметры доступны для Тюнинга
# когда C большой, он будет корректно классифицировать все точки данных, то есть оверфитить модель
# при C->inf будет соответствовать hard-margin svm
# coef0 контролирует, насколько на модель влияют полиномы высокой степени.
# gammaо пределяет, какое влияние оказывает отдельный учебный пример. 
#                  Чем больше gamma, тем ближе другие примеры должны быть затронуты.

parameters = {
    'classifier__degree':[2,3],
    'classifier__C':[0.0005, 0.005, 0.001, 0.01],
    'classifier__gamma':[1000, 300, 100, 10],
    'classifier__coef0': [0,1,10]
}
grid_cv_PSVC = GridSearchCV(model_poly_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', return_train_score=False)
# random_cv_PSVC = RandomizedSearchCV(model_poly_SVC, parameters, scoring=scorer, cv=cv, n_jobs=-1, refit='accuracy', random_state=42, n_iter=10)
%%time
grid_cv_PSVC.fit(X, y)
result_PSVC_cv_search = grid_cv_PSVC.cv_results_
get_result_from_grid_cv(result_PSVC_cv_search, grid_cv_PSVC)
# SPCA, t-sne
# спецкурс по обработке изображений ВМК МГУ Конушин
lda = LDA()
model_LDA = Pipeline(steps = [('scaling', MinMaxScaler()), ('classifier', lda)])
model_LDA.get_params().keys() # Глянем какие параметры доступны для Тюнинга
%%time
lda_scoring = cross_val_score(model_LDA, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy mean: {}, std: {}'.format(lda_scoring.mean(), lda_scoring.std()))
# -
# -
