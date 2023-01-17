# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# Загружаем данные
df = pd.read_csv('../input/vodafone-subset-3.csv')
df_2 = df[['target','ROUM','AVG_ARPU','car','gender','ecommerce_score','gas_stations_sms','phone_value','calls_duration_in_weekdays','calls_duration_out_weekdays','calls_count_in_weekends','calls_count_out_weekends']]
df_2
y = df_2['target'] 
X = df_2.drop('target',axis = 1)#датасет без 'target'
X
X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                     y , 
                                                      test_size=0.25, 
                                                      random_state=123)
tree = DecisionTreeClassifier(max_depth=10) # max_depth --- один из гиперпараметров дерева
tree.fit(X_train, y_train)
y_pred = tree.predict(X_valid)
print('Качество модели:', accuracy_score(y_valid, y_pred))
kf = KFold(n_splits=5, shuffle=True, random_state=12) # n_splits играет роль K
tree = DecisionTreeClassifier(max_depth=10)
scores = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')
print('Массив значений метрики:', scores)
print('Средняя метрика на кросс-валидации:', np.mean(scores))
tree_params={'max_depth': np.arange(2, 15)} # словарь параметров (ключ: набор возможных значений)

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
# Смотрим лучшие значения параметров
print(tree_grid.best_params_)

# Лучшая модель
print(tree_grid.best_estimator_)
# Результаты кросс-валидации в виде таблицы
pd.DataFrame(tree_grid.cv_results_).T
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision:', precision_score(y_valid, y_pred,average='macro'))
print('Recall:', recall_score(y_valid, y_pred,average='macro'))
print('F1 score:', f1_score(y_valid, y_pred,average='macro'))
#Обучение случайного леса с max_depth=5 (оптимальная глубина)
rf = RandomForestClassifier(n_estimators=100, random_state=2019, max_depth=5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)

accuracy_score(y_valid, y_pred)
#Обучение случайного леса (max_depth по умолчанию)
rf = RandomForestClassifier(n_estimators=100, random_state=2019)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, y_pred))
rf = RandomForestClassifier(n_estimators=100, random_state=2019)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, y_pred))
from sklearn.model_selection import GridSearchCV

rf_params={'n_estimators': np.arange(10, 110, 10)} # словарь параметров (ключ: набор возможных значений) от 10-200 с шагом 10

tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
# Смотрим лучшие значения параметров
print(tree_grid.best_params_)

# Лучшая модель
print(tree_grid.best_estimator_)
# Результаты кросс-валидации в виде таблицы
pd.DataFrame(tree_grid.cv_results_).T
import matplotlib.pyplot as plt
results_df = pd.DataFrame(tree_grid.cv_results_)
plt.plot(results_df['param_n_estimators'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('n_estimators')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
from sklearn.model_selection import GridSearchCV

rf_params={'max_features': np.arange(1,10)} 

tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
# Смотрим лучшие значения параметров
print(tree_grid.best_params_)

# Лучшая модель
print(tree_grid.best_estimator_)

print(tree_grid.best_score_)
# Результаты кросс-валидации в виде таблицы
pd.DataFrame(tree_grid.cv_results_).T
import matplotlib.pyplot as plt
results_df = pd.DataFrame(tree_grid.cv_results_)
plt.plot(results_df['param_max_features'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('max_features')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
from sklearn.model_selection import GridSearchCV

rf_params={'max_depth': np.arange(2,11)}

tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
# Смотрим лучшие значения параметров
print(tree_grid.best_params_)

# Лучшая модель
print(tree_grid.best_estimator_)
# Результаты кросс-валидации в виде таблицы
pd.DataFrame(tree_grid.cv_results_).T
import matplotlib.pyplot as plt
results_df = pd.DataFrame(tree_grid.cv_results_)
plt.plot(results_df['param_max_depth'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('max_max_depth')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
from sklearn.model_selection import GridSearchCV

rf_params={'min_samples_leaf': np.arange(3,10,2)}

tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
# Смотрим лучшие значения параметров
print(tree_grid.best_params_)

# Лучшая модель
print(tree_grid.best_estimator_)

print(tree_grid.best_score_)
# Результаты кросс-валидации в виде таблицы
pd.DataFrame(tree_grid.cv_results_).T
import matplotlib.pyplot as plt
results_df = pd.DataFrame(tree_grid.cv_results_)
plt.plot(results_df['param_min_samples_leaf'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('min_samples_leaf')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
from sklearn.model_selection import GridSearchCV

rf_params={'min_samples_leaf': np.arange(3,10,2),'max_depth': np.arange(2,11),'n_estimators': np.arange(10,70,10)}

tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
# Смотрим лучшие значения параметров
print(tree_grid.best_params_)

# Лучшая модель
print(tree_grid.best_estimator_)

print(tree_grid.best_score_)
# Результаты кросс-валидации в виде таблицы
pd.DataFrame(tree_grid.cv_results_).T
best_rf = tree_grid.best_estimator_
best_rf


import matplotlib.pyplot as plt

features = dict(zip(range(len(X.columns)), X.columns))

# Важность признаков
importances = best_rf.feature_importances_

indices = np.argsort(importances)[::-1]
# Plot the feature importancies of the forest
num_to_plot = max(1, len(X.columns))
feature_indices = [ind for ind in indices[:num_to_plot]]

# Print the feature ranking
print("Feature ranking:")

for f in range(num_to_plot):
    print(f+1, features[feature_indices[f]], importances[indices[f]])

plt.figure(figsize=(15,5))
plt.title("Feature importances")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision:', precision_score(y_valid, y_pred,average='macro'))
print('Recall:', recall_score(y_valid, y_pred,average='macro'))
print('F1 score:', f1_score(y_valid, y_pred,average='macro'))
import seaborn as sns
sns.heatmap(pd.crosstab(df_2['phone_value'], df_2['target']))
import seaborn as sns
sns.catplot(x='target', data=df_2, kind='count', col='phone_value', col_wrap=2)