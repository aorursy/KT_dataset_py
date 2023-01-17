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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from stop_words import get_stop_words
from sklearn.preprocessing import StandardScaler
# Загружаем данные
df = pd.read_csv('../input/vodafone-subset-3.csv')
df_2 = df[['target','ROUM','AVG_ARPU','car','gender','ecommerce_score','gas_stations_sms','phone_value','calls_duration_in_weekdays','calls_duration_out_weekdays','calls_count_in_weekends','calls_count_out_weekends']]
df_2
y = df_2['target'] 
X = df_2.drop('target',axis = 1)#датасет без 'target'
X.head()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, y_pred))
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
print(confusion_matrix(y_valid, y_pred))
plot_confusion_matrix(log_reg, X_valid, y_valid,values_format='5g')
plt.show()
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)
from sklearn.model_selection import GridSearchCV

log_reg = LogisticRegression(solver='liblinear', penalty='l2')

C_values = {'C': np.logspace(-3, 3, 10)}
logreg_grid = GridSearchCV(log_reg, C_values, cv=5, scoring='f1_macro')
logreg_grid.fit(X_train, y_train)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)
accuracy_score(y_valid, y_pred)
print(logreg_grid.best_params_)
print(logreg_grid.best_score_)
# Лучшая модель
print(logreg_grid.best_estimator_)
pd.DataFrame(logreg_grid.cv_results_).T
results_df = pd.DataFrame(logreg_grid.cv_results_)
plt.plot(results_df['param_C'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('C')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
y.value_counts(normalize=True)
y.value_counts(normalize=True).plot(kind='barh')
plt.show()
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
print(confusion_matrix(y_valid, y_pred))
plot_confusion_matrix(log_reg, X_valid, y_valid, values_format='5g')
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision:', precision_score(y_valid, y_pred,average='macro'))
print('Recall:', recall_score(y_valid, y_pred,average='macro'))
print('F1 score:', f1_score(y_valid, y_pred,average='macro'))
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)
log_reg = LogisticRegression(solver='liblinear', penalty='l1')

C_values = {'C': np.logspace(-3, 3, 10)}
logreg_grid = GridSearchCV(log_reg, C_values, cv=5, scoring='f1_macro')
logreg_grid.fit(X_train, y_train)
print(logreg_grid.best_params_)
print(logreg_grid.best_score_)
results_df = pd.DataFrame(logreg_grid.cv_results_)
plt.plot(results_df['param_C'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('C')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
y_pred = logreg_grid.best_estimator_.predict(X_valid)
print(confusion_matrix(y_valid, y_pred))
print('F1 score valid:', f1_score(y_valid, y_pred,average='macro'))