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

import seaborn as sns

import matplotlib.pyplot as plt
#Загрузка данных

df = pd.read_csv("/kaggle/input/bank-marketing-dataset/bank.csv")

df.head()
# Статистика по числовым признакам

df.describe().T
df.info()
df['deposit']
df['deposit'].hist()
new_values = {'yes':  1, 'no': 0} 

df['deposit1'] = df['deposit'].map(new_values)
df['housing1'] = df['housing'].map(new_values)
df['loan1'] = df['loan'].map(new_values)
df['default1'] = df['default'].map(new_values)

df['contact'].value_counts()
new_values_1 = {'cellular':  1, 'unknown': 0,'telephone' : 1 } 



df['contact1'] = df['contact'].map(new_values_1)

df['contact1'].value_counts()
new_values_2 = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec':12}

df['month1'] = df['month'].map(new_values_2)
df = df.drop(['deposit','housing','loan','default','contact', 'month'], axis = 1)
df = pd.get_dummies(df, columns=['marital','poutcome','education'])

df.head().T
from sklearn.model_selection import train_test_split



X = df.drop('deposit1', axis=1).drop('job',axis = 1)

y = df['deposit1'] 





# Разделение

# test_size --- доля исходных данных, которую оставляем для валидации

# random_state --- произвольное целое число, для воспроизводимости случайных результатов



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)

X.info()
print(X_train.shape, y_train.shape)
print( X_valid.shape, y_valid.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_valid)
from sklearn.metrics import accuracy_score



knn.score(X_valid, y_valid) #вычисляем метрику качества
print('Качество модели:', accuracy_score(y_pred, y_valid)) #другой способ для вычисления метрики
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=42) # n_splits играет роль K

scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')

print('Массив значений метрики:', scores)
from sklearn.model_selection import GridSearchCV



knn_params={'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn,knn_params, scoring = 'accuracy', cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_ # Лучшая модель
knn_grid.best_params_ # Лучшие значения параметров
knn_grid.best_score_
# Результаты кросс-валидации в виде таблицы

pd.DataFrame(knn_grid.cv_results_).T
# Валидационная кривая

import matplotlib.pyplot as plt



results_df = pd.DataFrame(knn_grid.cv_results_)

plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])



plt.xlabel('Number of neighbors')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
p_params = {'p': np.linspace(1,10,200)}

knn = KNeighborsClassifier(n_neighbors=27, weights = 'distance', n_jobs = -1)

cv = GridSearchCV(knn, p_params, cv = kf, scoring='accuracy', verbose = 100)

cv.fit(X,y)


cv_result = pd.DataFrame(cv.cv_results_)

cv_result

cv.best_estimator_
cv.best_score_
cv.best_params_
from sklearn.neighbors import RadiusNeighborsClassifier

def predy(r):

    neigh = RadiusNeighborsClassifier(radius=r)

    neigh.fit(X, y)

    y_pred = neigh.predict(X_valid)

    return y_pred



accuracy_score(predy(1), y_valid)
t = np.arange(1,100)

ac = [accuracy_score(predy(ts),y_valid) for ts in t  ]

plt.plot(t,ac)
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid()

nc.fit(X_train, y_train)

NearestCentroid()

y_pred_nc = nc.predict(X_valid)

accuracy_score(y_valid, y_pred_nc)