import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 8)}); 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/adult-income-dataset/adult.csv')
df.head(10).T
# Статистика по числовым признакам
df.describe().T
df.info()
df['income']
df['income'].value_counts().plot(kind='bar', color='black', figsize=(5,5))
plt.figure()
df['income']=df['income'].map({ '>50K': 1, '<=50K': 0})
df.head(10).T
new_df = pd.get_dummies(df, columns=['workclass', 'occupation', 'native-country']) 
new_df.head()
df = df[df["workclass"] != "?"]
df = df[df["occupation"] != "?"]
df = df[df["native-country"] != "?"]
df.head()
df1 =['workclass', 'race', 'education','marital-status', 'occupation', 'relationship', 'gender','native-country',
      'income'] 
for i in df1:
    unique_value, index = np.unique(df[i], return_inverse=True) 
    df[i] = index
df.head(10).T
from sklearn.model_selection import train_test_split
X = df.drop(['income'], axis=1)
y = df['income']
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=0.3, random_state=2019) # random_state=2019 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
neighbors = KNeighborsClassifier(n_neighbors=5)
neighbors.fit(X_train, y_train)
y_pred = neighbors.predict(X_valid)
print(accuracy_score(y_valid, y_pred))
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
neighbors = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(neighbors, X, y, cv=kf, scoring='accuracy')
print('Массив значений метрики:', scores)
print('Средняя метрика на кросс-валидации:', np.mean(scores))
from sklearn.model_selection import GridSearchCV
neighbors_params = {'n_neighbors': np.arange(1, 50)} # словарь параметров (ключ: набор возможных значений)
neighbors_grid = GridSearchCV(neighbors, neighbors_params, cv=kf, scoring='accuracy')  # кросс-валидация по 5 блокам
neighbors_grid.fit(X_train, y_train)

# Смотрим лучшие значения параметров
print(neighbors_grid.best_params_)

# Лучшая модель
print(neighbors_grid.best_estimator_)
#Оценка качества равна:
neighbors_grid.best_score_ 
# Результаты кросс-валидации в виде таблицы
pd.DataFrame(neighbors_grid.cv_results_).T
# Рисуем валидационную кривую
# По оси х --- значения гиперпараметров 
# По оси y --- значения метрики

results_df = pd.DataFrame(neighbors_grid.cv_results_)
plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])
plt.figure()
p_params = {'p': np.linspace(1,10,200)}
neighbors = KNeighborsClassifier(n_neighbors=26, weights = 'distance', n_jobs = -1)
cv = GridSearchCV(neighbors, p_params, cv = kf, scoring='accuracy', verbose = 100)
cv.fit(X,y)
cv.best_params_
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid()
nc.fit(X_train, y_train)

nc.score(X_valid, y_valid)