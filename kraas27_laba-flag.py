import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from pylab import rcParams
data = pd.read_csv('../input/flag.data', names=['name', 'landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes',

              'colours', 'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'circles',

              'crosses', 'saltires', 'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 

              'text', 'topleft', 'botright'])
data.head()
data.info()
data.describe()
rcParams['figure.figsize'] = (10, 8)

sns.heatmap(data.corr()) # проверим признаки на корреляцию
# Сильную корреляцию между признаками не наблюдаем
data.religion.hist()
# По целевой переменной видим что различные религии представлены разным количеством примперов, потому будем

# делить их с помощью ShuffleSplit
# выделим категориальные признаки

data_cat = data.dtypes[data.dtypes==object].index.tolist()
for i in data[data_cat]:

    print (i, len(data[i].value_counts().index))
# Имя страны никакой полезной информации не несет, сделаем ее просто индексом

# так как остальных значений не много используем dummies
# Так как признаковое пространство совсем небольшое, то уменьшать его не вижу необходимости 
data.index=data.name

data = data.drop('name', axis=1)
data = pd.get_dummies(data, columns=['mainhue', 'topleft', 'botright'])
data_target = data['religion'].copy()
data = data.drop('religion', axis=1)
# Попробуем добавить дополнительный признак полученный с помощью Kmeans
from sklearn.cluster import KMeans
k_inertia = []

ks = range(1,11)



for k in ks:

    clf_km = KMeans(n_clusters=k)

    clusters_km = clf_km.fit_predict(data, )

    k_inertia.append(clf_km.inertia_/100)
rcParams['figure.figsize'] = (12,6)

plt.plot(ks, k_inertia)
# По графику видим что оптимальное разделение будет на 2-ух и трёх кластерах, больше брать нет смысла. 

# Возьмём 3
km = KMeans(n_clusters=3)

km.fit(data)

clusters = km.predict(data)

data['cluster'] = clusters
from sklearn.model_selection import StratifiedShuffleSplit



splitter = StratifiedShuffleSplit(n_splits = 5, test_size = 0.25, random_state=27)



for train_index, test_index in splitter.split(data, data_target):

    X_train = data.iloc[train_index]

    X_valid = data.iloc[test_index]

    y_train = data_target.iloc[train_index]

    y_valid = data_target.iloc[test_index]
# попробуем добавить малых классов
col = X_train
from collections import Counter

from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(k_neighbors=2, ratio={3:30, 4:30, 6:30, 7:30},

                                 random_state=27).fit_resample(X_train, y_train)

print(sorted(Counter(y_resampled).items()))
X_train = pd.DataFrame(X_resampled, columns=col.columns)
y_train = pd.Series(y_resampled)
y_train.hist()
# Попробуем сделать класификацию на RandomForest

from sklearn.ensemble import RandomForestClassifier

# Заодно сделаем Kfold и подберем оптимальные параметры на Grigsearch

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_jobs=-1, random_state=27)
criterion = ['gini', 'entropy']

max_depth = [5, 7, 9, 11, 13, 15]

min_samples_leaf = [3, 5, 8, 10, 15, 25]

param_grid = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'criterion': criterion}

rf_gs = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
rf_gs.fit(X_train, y_train)
print (rf_gs.best_score_)

print (rf_gs.best_params_)

rf_gs_best = rf_gs.best_estimator_
y_pred = rf_gs_best.predict(X_valid)
from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
# Выводим важность признаков 

imp = pd.Series(rf_gs_best.feature_importances_, index=X_train.columns)

imp.sort_values(ascending=False)
# Лес справился не очень хорошо, посмотрим как справится линейный метод

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
penal = ['l1', 'l2']

c = [0.01, 0.05, 0.1, 0.5, 1, 5]

param_grid = {'penalty': penal, 'C': c}

lr_gs = GridSearchCV(log_reg, param_grid, cv=10, n_jobs=-1, scoring='accuracy')
lr_gs.fit(X_train, y_train)
print (lr_gs.best_params_)

print (lr_gs.best_score_)

log_reg = lr_gs.best_estimator_
# Немного хуже... ещё бы хорошо проверить xgboost (только он долго обучается, не успею настроить)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, recall_score
accuracy_score(y_valid, y_pred)
f1_score(y_valid, y_pred, average='weighted')
recall_score(y_valid, y_pred, average='weighted')
# precision_recall_curve?
confusion_matrix = pd.crosstab(y_valid, y_pred, rownames=['Actual'],

                               colnames=['Predicted'], margins = True)



sns.heatmap(confusion_matrix, annot=True)