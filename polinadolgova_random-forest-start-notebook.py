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
X_train = pd.read_csv('../input/hsemath2020flights/flights_train.csv')
X_test = pd.read_csv('../input/hsemath2020flights/flights_test.csv')
X_train.head(5)
y_train = pd.DataFrame()
y_train['target'] = np.where(X_train['dep_delayed_15min'], 1, 0)
del X_train['dep_delayed_15min']
def month(x: str):
    return int(x.split('-')[1])

X_train['MONTH'] = X_train['DATE'].apply(month)
X_test['MONTH'] = X_test['DATE'].apply(month)
X_train = X_train[['MONTH', 'DISTANCE', 'DEPARTURE_TIME']]
X_test = X_test[['MONTH', 'DISTANCE', 'DEPARTURE_TIME']]
X_train.head(5)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
X_train[['DISTANCE', "DEPARTURE_TIME"]].describe()
plt.figure(figsize=(5, 4))
sns.distplot(X_train['DISTANCE'], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.title("распределение признака distance")
plt.ylabel('частота значения')

plt.figure(figsize=(5, 4))
sns.distplot(X_train['DEPARTURE_TIME'], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.title("распределение признака departure_time")
plt.ylabel('частота значения')
plt.figure(figsize=(5, 4))
sns.distplot(X_train['MONTH'], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.title("распределение признака month")
plt.ylabel('частота значения')
X_train['dep_delayed_15min'] = y_train
correlation = X_train.corr()
correlation
sns.heatmap(correlation, cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,annot=True, annot_kws={"size": 8}, square=True);
mean_target_by_month = X_train.groupby('MONTH').mean()['dep_delayed_15min']

columns = np.arange(1, 13)
data = np.array(mean_target_by_month)

plt.bar(columns, data, align='center', alpha=0.8, color = 'gold')
plt.xticks(columns)
plt.ylabel('Среднее dep_delayed_15min')
plt.title('Среднее dep_delayed_15min для каждого месяца')

plt.show()
del X_train['dep_delayed_15min']
from sklearn import model_selection
train, test, ytrain, ytest = model_selection.train_test_split(X_train, y_train, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
random_forest = RandomForestClassifier(random_state=0, n_jobs = -1, criterion= 'entropy', max_features='auto')

#выбираем диапазон перебираемых параметров
all_params = grid = np.arange(40, 110, 10)
#перебираем параметры с помощью GridSearchCV, количество фолдов при cv = 4
grid = {'n_estimators' : all_params}
grid_search = GridSearchCV(random_forest, grid, scoring = 'roc_auc', cv = 4)
grid_search.fit(np.array(train), np.array(ytrain).ravel())

print("Лучший параметр:{} \n ".format(grid_search.best_params_))
best_model = grid_search.best_estimator_
predict_proba_test = best_model.predict_proba(test)

#считаем roc auc на нашей тестовой выборке
roc_auc_test = metrics.roc_auc_score(ytest,predict_proba_test[:,1])
# для обучщающей выборки
roc_auc_train = grid_search.best_score_

print('ROC AUC: для train:{}, для test: {}'.format(roc_auc_train, roc_auc_test))
mean_test_score = grid_search.cv_results_['mean_test_score']

plt.figure(figsize=(5, 4))
plt.plot(all_params, mean_test_score, marker='o', color = 'orange')
plt.ylabel('roc_auc')
plt.xlabel('n_estimators')
plt.title('среднее при кросс-валидации качество от n_estimators')
plt.show()
final_prediction = best_model.predict_proba(X_test)[:, 1]

dfpred = pd.DataFrame()
dfpred['Id'] = np.arange(len(final_prediction))
dfpred['dep_delayed_15min'] = final_prediction
dfpred.to_csv('predict.csv', index = False)