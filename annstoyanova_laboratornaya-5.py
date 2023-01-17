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
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import warnings
warnings.filterwarnings('ignore')
import os
import re
import numpy as np
import pandas as pd
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df=pd.read_csv("/kaggle/input/vodafone/vodafone-subset-1.csv")
df.head()


df1=df.drop('target', axis = 1)
#.select_dtypes(exclude=['object'])
target=df[['target']]
target['target'].value_counts().plot(kind='bar');
df1= df[['SCORING', 'AVG_ARPU', 'ROUM', 'calls_count_in_weekdays', 'DATA_VOLUME_WEEKDAYS', 'car','uklon', 'gender','Oblast_post_HOME','Raion_post_HOME','City_post_HOME','banks_sms_count','telegram_count','linkedin_count','skype_count','sim_count','intagram_count','whatsapp_volume']]
df1.rename(columns={'SCORING': 'Уровень дохода', 
                     'AVG_ARPU': 'Стоимость услуг',
                     'ROUM': 'Факт поездок заграницу',
                     'car': 'Наличие машины',
                     'calls_count_in_weekdays': 'Кол-во использование телефона в будние',
                     'DATA_VOLUME_WEEKDAYS': 'Кол-во использованного трафика',
                     'uklon':'ипользование такси УКЛОН',
                     'gender':'пол',
                   'Oblast_post_HOME':'Область_дом',
                   'Raion_post_HOME':'Район_дом',
                    'City_post_HOME':'Город_дом',
                    'banks_sms_count':'Количество входящих сообщений от всех банков',
                    'telegram_count':'кол-во трафика телеграм',
                    'linkedin_count':'кол-во трафика Линкдн',
                    'skype_count':'кол-во трафика Скайп',
                    'sim_count':'кол-во симок',
                    'intagram_count':'кол-во трафика инстаграм',
                    'whatsapp_volume':'кол-во трафика вотсап'
                    
                   }, inplace=True)
df1
df1.info()

label_encoder = LabelEncoder()

obl = pd.Series(label_encoder.fit_transform(df1['Область_дом']))
obl.value_counts().plot.barh()
print(dict(enumerate(label_encoder.classes_)))
label_encoder = LabelEncoder()
rayon=pd.Series(label_encoder.fit_transform(df1['Район_дом']))
rayon.value_counts().plot.barh()
print(dict(enumerate(label_encoder.classes_)))

label_encoder = LabelEncoder()
gor=pd.Series(label_encoder.fit_transform(df1['Город_дом']))
gor
categorical_columns = df1.columns[df1.dtypes == 'object'].union(['Район_дом'])
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df1[column])
df1.head()
df1['Область_дом'] = obl
df1['Район_дом'] = rayon
df1['Город_дом'] = gor
df1['индекс уровня дохода'] = df1['Уровень дохода'].map({'HIGH':6,
                                               'HIGH_MEDIUM':5,
                                               'MEDIUM':4,
                                               'LOW':3,
                                               'VERY LOW':2,
                                               '0':1})


df1=df1.drop('Уровень дохода',axis=1)
df1.head()
#X=df.drop('target',axis=1)
#y=df['target']
X=df1.drop('ипользование такси УКЛОН',axis=1)
y=target
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                      y,
                                                      test_size=0.3,
                                                      random_state=2020)
df1.dropna().shape
X.info()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=2019,max_depth=2)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, y_pred))

rf = RandomForestClassifier(n_estimators=100, random_state=2019)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, y_pred))


kf = KFold(n_splits=5, shuffle=True, random_state=12) # n_splits играет роль K
tree = DecisionTreeClassifier(max_depth=10)
scores = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')
print('Массив значений метрики:', scores)
print('Средняя метрика на кросс-валидации:', np.mean(scores))
import matplotlib.pyplot as plt

features = dict(zip(range(len(df.columns)-1), df.columns[:-1]))

# Важность признаков
importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]
# Plot the feature importancies of the forest
num_to_plot = max(10, len(df.columns[:-1]))
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
from sklearn.model_selection import GridSearchCV

rf_params={'n_estimators': np.arange(10, 210, 10)} # словарь параметров (ключ: набор возможных значений) от 10-200 с шагом 10

tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
print('Лучшие значения параметров:' ,tree_grid.best_params_)

print('Лучшая модель:' ,tree_grid.best_estimator_)
print(tree_grid.best_score_)
pd.DataFrame(tree_grid.cv_results_).T
df4=pd.DataFrame(tree_grid.cv_results_)
plt.plot(df4['param_n_estimators'], df4['mean_test_score'])

from sklearn.model_selection import GridSearchCV

rf_params={'max_features': np.arange(1,10)} 

tree_grid=GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
print('Лучшие значения параметров:' ,tree_grid.best_params_)


print('Лучшая модель:' ,tree_grid.best_estimator_)
print(tree_grid.best_score_)
pd.DataFrame(tree_grid.cv_results_).T
df5=pd.DataFrame(tree_grid.cv_results_)
plt.plot(df5['param_max_features'], df5['mean_test_score'])
rf_params={'max_depth': np.arange(2,11)}
tree_grid=GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
tree_grid.fit(X_train, y_train)
print('Лучшие значения параметров:' ,tree_grid.best_params_)

print('Лучшая модель:' ,tree_grid.best_estimator_)
print(tree_grid.best_score_)
pd.DataFrame(tree_grid.cv_results_).T
df6=pd.DataFrame(tree_grid.cv_results_)
plt.plot(df6['param_max_depth'], df6['mean_test_score'])

rf_params={'min_samples_leaf': np.arange(3,10,2)}
tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') 
tree_grid.fit(X_train, y_train)
print('Лучшая модель:' ,tree_grid.best_estimator_)
print('Лучшие значения параметров:' ,tree_grid.best_params_)
print(tree_grid.best_score_)
pd.DataFrame(tree_grid.cv_results_).T
df7=pd.DataFrame(tree_grid.cv_results_)
plt.plot(df7['param_min_samples_leaf'], df7['mean_test_score'])

rf_params={'min_samples_leaf': np.arange(3,10,2),'max_depth': np.arange(2,11),'n_estimators': np.arange(10,120,10)}

tree_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') 
tree_grid.fit(X_train, y_train)
print('Лучшая модель:' ,tree_grid.best_estimator_)
print('Лучшие значения параметров:' ,tree_grid.best_params_)
print(tree_grid.best_score_)
pd.DataFrame(tree_grid.cv_results_).T
best_rf = tree_grid.best_estimator_
features = dict(zip(range(len(df1.columns)-1), df1.columns[:-1]))

# Важность признаков
importances = best_rf.feature_importances_

indices = np.argsort(importances)[::-1]
# Plot the feature importancies of the forest
num_to_plot = max(10, len(df1.columns[:-1]))
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