import pandas as pd
from pandas import Series
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/sf-dst-scoring/train.csv')
test= pd.read_csv('/kaggle/input/sf-dst-scoring/test.csv')
#train = pd.read_csv('./train_4.csv')
#test = pd.read_csv('./test_4.csv')

train.info()
test.info()
train.sample(10)
train.isnull().sum()
test.isnull().sum()
train['education'].fillna(method='ffill', inplace=True)
train.info()
#В тестовом наборе строки с пропусками удалим
test.dropna(inplace = True)
test.info()
#Преобразуем даты
train.app_date = pd.to_datetime(train.app_date)
test.app_date = pd.to_datetime(train.app_date)
print(train.app_date.sample(2))
print(test.app_date.sample(2))
# Выясняем начало и конец периода тренировочного датасета
start = train.app_date.min()
end = train.app_date.max()
start,end
# Выясняем начало и конец периода тестового датасета
start = test.app_date.min()
end = test.app_date.max()
start,end
#Добавим новые признаки
# Количество дней от старта заявки 
train['td'] = train.app_date - train.app_date.min()
train['td'] = train['td'].apply(lambda x: str(x).split()[0])
train['td'] = train['td'].astype(int)
test['td'] = test.app_date - test.app_date.min()
test['td'] = test['td'].apply(lambda x: str(x).split()[0])
test['td'] = test['td'].astype(int)
#Месяц подачи заявки
train['app_date_month'] = train.app_date.dt.month
test['app_date_month'] = test.app_date.dt.month
train.head()
test.head()
#Полученные признаки сгруппируем в три категории по типу их обработки (категориальные, бинарные и числовые)
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
cat_cols = ['app_date_month', 'education', 'home_address', 'work_address', 'sna', 'first_time']
num_cols = ['td', 'age', 'decline_app_cnt', 'score_bki', 'bki_request_cnt', 'region_rating', 'income']
# Обработаем числовые переменные

for i in num_cols:
    plt.figure()
    sns.distplot(train[i][train[i] > 0].dropna(), kde = False, rug=False)
    plt.title(i)
    plt.show()
#Прологарифмируем некоторые переменные
num_cols_log = ['decline_app_cnt', 'bki_request_cnt', 'income']
for i in num_cols_log:
    train[i] = np.log(train[i] + 1)
    plt.figure(figsize=(10,6))
    sns.distplot(train[i][train[i] > 0].dropna(), kde = False, rug=False)
    plt.show()
#Проделаем то же самое с тестовым сетом
for i in num_cols_log:
    test[i] = np.log(test[i] + 1)
    plt.figure(figsize=(10,6))
    sns.distplot(test[i][test[i] > 0].dropna(), kde = False, rug=False)
    plt.show()
sns.heatmap(train.corr().abs(), vmin=0, vmax=1)
#Определяем, какой числовой признак самый важный
imp_num = Series(f_classif(train[num_cols], train['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh')
sns.boxplot(x=train.default, y=train.score_bki)
sns.boxplot(x=train.default, y=train.age)
# Для бинарных  и категориальных признаков мы будем использовать LabelEncoder

label_encoder = LabelEncoder()

for column in bin_cols:
    train[column] = label_encoder.fit_transform(train[column])
    test[column] = label_encoder.fit_transform(test[column])
for column in cat_cols:
    train[column] = label_encoder.fit_transform(train[column])
    test[column] = label_encoder.fit_transform(test[column])
# убедимся в преобразовании    
display(train.head())

#Преобразуем категориальные переменные при помощи OneHotEncoder
x_cat = OneHotEncoder(sparse = False).fit_transform(train[cat_cols].values)
x_cat_test = OneHotEncoder(sparse = False).fit_transform(test[cat_cols].values)

print(x_cat.shape)
print(x_cat_test.shape)
## Значимость категориальных признаков

imp_cat = pd.Series(mutual_info_classif(train[bin_cols + cat_cols],
                                        train['default'], discrete_features =True),
                    index = bin_cols + cat_cols)
imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh')
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)

x_tr = poly.fit_transform(train[num_cols].values)
x_test = poly.fit_transform(test[num_cols].values)
x_num = StandardScaler().fit_transform(x_tr)
x_num_test = StandardScaler().fit_transform(x_test)
print(x_num)
print(x_num_test)
# Объединяем
X = np.hstack([x_num, train[bin_cols].values, x_cat])
Y = train['default'].values

id_test = test.client_id
test = np.hstack([x_num_test, test[bin_cols].values, x_cat_test])
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV

# Добавим типы регуляризации
penalty = ['l1', 'l2']

# Зададим ограничения для параметра регуляризации
C = np.logspace(1, 4, 8)
# Создадим гиперпараметры
hyperparameters = dict(C=C, penalty=penalty)

model = LogisticRegression()
model.fit(X_train, y_train)

# Создаем сетку поиска с использованием 5-кратной перекрестной проверки
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Лучшее C:', best_model.best_estimator_.get_params()['C'])
lgr = LogisticRegression(penalty = 'l2', C=10000, max_iter=500)
lgr.fit(X_train, y_train)
probs = lgr.predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
lgr = LogisticRegression(penalty = 'l2', C=10000, max_iter=500)
lgr.fit(X, Y)
probs = lgr.predict_proba(test)
probs = probs[:,1]
my_submission = pd.DataFrame({'client_id': id_test, 
                            'default': probs})
my_submission.to_csv('submission.csv', index=False)

my_submission