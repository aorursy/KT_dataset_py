import numpy as np # работа с датасетом

import pandas as pd # математическая библиотека

from pandas import read_csv, DataFrame, Series # чтение данных

from sklearn.model_selection import train_test_split, KFold, cross_val_score # подготовка дынных и анализ результатов

import matplotlib.pyplot as plt # построение графика

from sklearn.preprocessing import LabelEncoder # манипуляции с данными



from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
data = read_csv('../input/german-credit-data-with-risk/german_credit_data_with_risk.csv')
data.shape
data.head(3)
data.tail(3)
data.drop(data.columns[0], inplace=True, axis=1)
data.head(3)
data.tail(3)
data.describe()
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']

numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']

print(categorical_columns)

print(numerical_columns)
data[categorical_columns].describe()
for c in categorical_columns:

    print(data[c].unique())
from pandas.tools.plotting import scatter_matrix

scatter_matrix(data, alpha=0.05, figsize=(20, 20));
data.corr()
data.count(axis=0)
data_top = read_csv('../input/german-credit-data-with-risk/german_credit_data_with_risk.csv')

data_new = read_csv('../input/german-credit-data-with-risk/german_credit_data_with_risk.csv')
data_describe = data_top.describe(include=[object])

for c in categorical_columns:

    data_top[c] = data_top[c].fillna(data_describe[c]['top'])
data_top.head(10)
data_describe = data_new.describe(include=[object])

for c in categorical_columns:

    data_new[c] = data_new[c].fillna('no_inf')
data_new.head(10)
binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]

nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]

print('Binary:', binary_columns)

print('Non binary:', nonbinary_columns)
def SetBinary(data):

    label = LabelEncoder()

    dicts = {}



    label.fit(data.Sex.drop_duplicates())

    dicts['Sex'] = list(label.classes_)

    data.Sex = label.transform(data.Sex)



    label.fit(data.Risk.drop_duplicates())

    dicts['Risk'] = list(label.classes_)

    data.Risk = label.transform(data.Risk)

    return data
data_new = SetBinary(data_new)

dara_top = SetBinary(data_top)
data_top.head(5)
data_new.head(5)
data_nonbinary_top = pd.get_dummies(data_top[nonbinary_columns])

print(data_nonbinary_top.columns)
data_nonbinary_new = pd.get_dummies(data_new[nonbinary_columns])

print(data_nonbinary_new.columns)
data_numerical_top = data_top[numerical_columns]

data_numerical_top = (data_numerical_top - data_numerical_top.mean()) / data_numerical_top.std()

data_numerical_top.describe()
data_numerical_new = data_new[numerical_columns]

data_numerical_new = (data_numerical_new - data_numerical_new.mean()) / data_numerical_new.std()

data_numerical_new.describe()
data_top = pd.concat((data_numerical_top, data_top[binary_columns], data_nonbinary_top), axis=1)

data_top = pd.DataFrame(data_top, dtype=float)

data_top.head(5)
data_new = pd.concat((data_numerical_new, data_new[binary_columns], data_nonbinary_new), axis=1)

data_new = pd.DataFrame(data_new, dtype=float)

data_new.head(5)
data_top.shape
data_new.shape
X_top = data_top.drop(('Risk'), axis=1)

y_top = data_top['Risk']



X_new = data_new.drop(('Risk'), axis=1)

y_new = data_new['Risk']
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y_top, test_size = 0.3, random_state = 11)

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size = 0.3, random_state = 11)
itog_val = {} #список для записи результатов работы методов
knn = KNeighborsClassifier()

knn.fit(X_train_top, y_train_top)
y_train_predict_top = knn.predict(X_train_top)

y_test_predict_top = knn.predict(X_test_top)



err_train = np.mean(y_train_top != y_train_predict_top)

err_test  = np.mean(y_test_top  != y_test_predict_top)

print('Train top error:', err_train)

print('Test top error', err_test)

itog_val['KNeighborsClassifierTop'] = np.mean(y_test_top == y_test_predict_top)
knn.fit(X_train_new, y_train_new)
y_train_predict_new = knn.predict(X_train_new)

y_test_predict_new = knn.predict(X_test_new)



err_train = np.mean(y_train_new != y_train_predict_new)

err_test  = np.mean(y_test_new  != y_test_predict_new)

print('Train new error:', err_train)

print('Test new error', err_test)



itog_val['KNeighborsClassifierNew'] = np.mean(y_test_new == y_test_predict_new)
svc = SVC()

svc.fit(X_train_top, y_train_top)



err_train = np.mean(y_train_top != svc.predict(X_train_top))

err_test  = np.mean(y_test_top  != svc.predict(X_test_top))

print('Train top error:', err_train)

print('Test top error', err_test)



itog_val['SVC_Top'] = np.mean(y_test_top == y_test_predict_top)
svc.fit(X_train_new, y_train_new)

y_train_predict_new = svc.predict(X_train_new)

y_test_predict_new = svc.predict(X_test_new)



err_train = np.mean(y_train_new != y_train_predict_new)

err_test  = np.mean(y_test_new  != y_test_predict_new)

print('Train new error:', err_train)

print('Test new error', err_test)



itog_val['SVC_New'] = np.mean(y_test_new == y_test_predict_new)
rf = RandomForestClassifier()



rf.fit(X_train_top, y_train_top)



err_train = np.mean(y_train_top != rf.predict(X_train_top))

err_test  = np.mean(y_test_top  != rf.predict(X_test_top))

print('Train top error:', err_train)

print('Test top error', err_test)



itog_val['RandomForest_Top'] = np.mean(y_test_top == y_test_predict_top)
rf.fit(X_train_new, y_train_new)

y_train_predict_new = rf.predict(X_train_new)

y_test_predict_new = rf.predict(X_test_new)



err_train = np.mean(y_train_new != y_train_predict_new)

err_test  = np.mean(y_test_new  != y_test_predict_new)

print('Train new error:', err_train)

print('Test new error', err_test)



itog_val['RandomForest_New'] = np.mean(y_test_new == y_test_predict_new)
DataFrame.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False, figsize=(10,6))