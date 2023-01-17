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
df = pd.read_csv('../input/adult-income-dataset/adult.csv')

df.head()
df.info()
df = df.drop(['race','gender','marital-status', 'education', 'fnlwgt'],axis=1)

df.head()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

dicts = {}



label.fit(df.relationship.drop_duplicates())

dicts['relationship'] = list(label.classes_)

df.relationship = label.transform(df.relationship) 



label.fit(df.workclass.drop_duplicates())

dicts['workclass'] = list(label.classes_)

df.workclass = label.transform(df.workclass)



label.fit(df.occupation.drop_duplicates())

dicts['occupation'] = list(label.classes_)

df.occupation = label.transform(df.occupation)



label.fit(df['native-country'].drop_duplicates())

dicts['native-country'] = list(label.classes_)

df['native-country'] = label.transform(df['native-country'])



label.fit(df.income.drop_duplicates())

dicts['income'] = list(label.classes_)

df.income = label.transform(df.income)



df.head(10)
from sklearn.model_selection import train_test_split



X = df.drop(['income'],  axis=1) 

y = df['income']



X_train, X_valid, y_train, y_valid = train_test_split(X, 

                                                      y, 

                                                      test_size=0.1, 

                                                      random_state=12)
df_dataframe = pd.DataFrame(X_train)

grr = pd.plotting.scatter_matrix(df_dataframe, 

                                 c=y_train, 

                                 figsize=(30, 30), 

                                 marker='o',

                                 hist_kwds={'bins': 20}, s=60, alpha=.8)
# Создаём представителя класса модели, задаём необходимые гиперпараметры

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train, y_train)
# Вычисляем метрику

y_pred = knn.predict(X_valid)

y_pred

knn.score(X_valid, y_valid)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X, y, 

                         cv=kf, scoring='accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)} # число соседей -- от 1 до 50

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=5) 

knn_grid.fit(X_train, y_train)
knn_grid.best_params_
knn_grid.best_score_
#from matplotlib.pyplot import plot

#plot(X, y)

import matplotlib.pyplot as plt

results_df = pd.DataFrame(knn_grid.cv_results_)

plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])



# Подписываем оси и график

plt.xlabel('n_neighbors')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
from sklearn.metrics import mean_squared_error, explained_variance_score

print('Mean squared error:', mean_squared_error(y_valid, y_pred))

print('Explained variance score:', explained_variance_score(y_valid, y_pred))