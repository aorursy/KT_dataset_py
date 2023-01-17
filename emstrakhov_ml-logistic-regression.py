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
np.random.seed(0)

import matplotlib.pyplot as plt

X = np.random.normal(loc=0.5, scale=0.25, size=(100, 2))

y = (X[:, 1] > X[:, 0]).astype('int') # разделяющая граница: y=x (биссектриса первой четверти)

plt.scatter(X[:, 0], X[:, 1], color=['red' if c==1 else 'blue' for c in y])

plt.xlabel('x1')

plt.ylabel('x2')

plt.show()
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X, y)



w = log_reg.coef_

bb = log_reg.intercept_



print(w, bb)
w1 = w[0][0]

w2 = w[0][1]

b = bb[0]



print('w1 = '+str(w1), '\nw2 = '+str(w2), '\nb = '+str(b))
plt.scatter(X[:, 0], X[:, 1], color=['red' if c==1 else 'blue' for c in y])

plt.plot(X[:, 0], -w1/w2*X[:, 0]-b/w2, color='green', linewidth=3, linestyle="dashed")



plt.xlabel('x1')

plt.ylabel('x2')

plt.show()
yy = ((X[:, 0]-0.5)**2 + (X[:, 1]-0.5)**2 > 0.25**2).astype('int')

plt.scatter(X[:, 0], X[:, 1], color=['red' if c==1 else 'blue' for c in yy])

plt.xlabel('x1')

plt.ylabel('x2')

plt.show()
log_reg = LogisticRegression()

log_reg.fit(X, yy)



w1 = log_reg.coef_[0][0]

w2 = log_reg.coef_[0][1]

b = log_reg.intercept_[0]



plt.scatter(X[:, 0], X[:, 1], color=['red' if c==1 else 'blue' for c in yy])

plt.plot(X[:, 0], -w1/w2*X[:, 0]-b/w2, color='green', linewidth=3, linestyle="dashed")

plt.xlabel('x1')

plt.ylabel('x2')

plt.show()
df = pd.read_csv('/kaggle/input/depression/b_depressed.csv')

df.head()
df.info()
df.nunique()
# Удалим пропуски

df_1 = df.dropna()



# Дропнем ненужные столбцы

df_2 = df_1.drop(['Survey_id', 'depressed'], axis=1)



# Переведём признаки "Номер виллы" и "Уровень образования" в бинарные 

# * мы не уверены на 100 %, что уровень образования ранговый, поэтому считаем его категориальным

df_3 = pd.get_dummies(df_2, columns=['Ville_id', 'education_level'])



# Масштабирование

col_names = df.columns.values # это имена всех столбцов

large_numbers = [col for col in col_names if df[col].mean() > 10000] # имена тех, у кого среднее > 10000

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_st = scaler.fit_transform(df_3[large_numbers])



# Переприсвоим старым колонкам новые

df_3[large_numbers] = X_st



df_3.head()
X = df_3

y = df_1['depressed']



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_valid)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_valid, y_pred))
y.value_counts(normalize=True)
y.value_counts(normalize=True).plot(kind='barh')

plt.show()
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

print(confusion_matrix(y_valid, y_pred))
plot_confusion_matrix(log_reg, X_valid, y_valid, values_format='5g')

plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision:', precision_score(y_valid, y_pred))

print('Recall:', recall_score(y_valid, y_pred))

print('F1 score:', f1_score(y_valid, y_pred))
from sklearn.model_selection import GridSearchCV



log_reg = LogisticRegression(solver='liblinear')



C_values = {'C': np.logspace(-3, 3, 10)}

logreg_grid = GridSearchCV(log_reg, C_values, cv=5, scoring='f1')

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
log_reg = LogisticRegression(solver='liblinear', penalty='l1')



C_values = {'C': np.logspace(-3, 3, 10)}

logreg_grid = GridSearchCV(log_reg, C_values, cv=5, scoring='f1')

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

print('F1 score valid:', f1_score(y_valid, y_pred))
# kNN (не помог)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)



y_pred = knn.predict(X_valid)

print(confusion_matrix(y_valid, y_pred))

print('F1 score valid:', f1_score(y_valid, y_pred))
knn_params = {'n_neighbors': np.arange(1, 50, 2)}

knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='f1')

knn_grid.fit(X_train, y_train)



y_pred = knn_grid.best_estimator_.predict(X_valid)

print(confusion_matrix(y_valid, y_pred))

print('F1 score valid:', f1_score(y_valid, y_pred))
# Random Forest (не помог)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)



y_pred = rf.predict(X_valid)

print(confusion_matrix(y_valid, y_pred))

print('F1 score valid:', f1_score(y_valid, y_pred))
# Искусственное добавление объектов класса 1



from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X_train, y_train)
# Проверим баланс

y_ros.value_counts()
# Логистическая регрессия с добавлением класса 1

logreg_ros = LogisticRegression(solver='liblinear')

logreg_ros.fit(X_ros, y_ros)

y_pred = logreg_ros.predict(X_valid)



print(confusion_matrix(y_valid, y_pred))

print('F1 score valid:', f1_score(y_valid, y_pred))
# Подбор гиперпараметров

logreg_params = {'C': np.logspace(-3, 3, 10), 'penalty': ['l2', 'l1']}

logreg_grid = GridSearchCV(logreg_ros, logreg_params, cv=5, scoring='f1')

logreg_grid.fit(X_ros, y_ros)



y_pred = logreg_grid.best_estimator_.predict(X_valid)

print(confusion_matrix(y_valid, y_pred))

print('F1 score valid:', f1_score(y_valid, y_pred))