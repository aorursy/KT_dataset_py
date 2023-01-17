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
train.head(2)
train.info(), test.info()
train.columns, test.columns
train.isnull().sum(), test.isnull().sum()
train['education'].value_counts().plot.barh()
test['education'].value_counts().plot.barh()
# fill NaN with the most frequent value

import collections

c_1 = collections.Counter(train['education'])
c_2 = collections.Counter(test['education'])

train['education'].fillna(c_1.most_common()[0][0], inplace=True)
test['education'].fillna(c_2.most_common()[0][0], inplace=True)
train.isnull().sum(), test.isnull().sum()
train.app_date.head(5), test.app_date.head(5)
# convert to datetime

train.app_date = pd.to_datetime(train.app_date)
test.app_date = pd.to_datetime(train.app_date)
print(train.app_date.head(2))
print(test.app_date.head(2))
current_date = pd.to_datetime('28JUL2020')

# Количество дней, прошедших со дня подачи заявки

train['days_passed'] = (current_date - train.app_date).dt.days
test['days_passed'] = (current_date - test.app_date).dt.days

# Месяц подачи заявки

train['app_date_month'] = train.app_date.dt.month
test['app_date_month'] = test.app_date.dt.month
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
cat_cols = ['app_date_month', 'education', 'home_address', 'work_address', 'sna', 'first_time']
num_cols = ['days_passed', 'age', 'decline_app_cnt', 'score_bki', 'bki_request_cnt', 'region_rating', 'income']
sns.countplot(train['default'])
train['default'].value_counts()
fig, axes = plt.subplots(2, 4, figsize=(25,12))
for col, i in zip(num_cols, range(7)):
    sns.distplot(train[col], kde=False, ax=axes.flat[i])
num_cols_log = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'days_passed']
for i in num_cols_log:
    train[i] = np.log(train[i] + 1)
    plt.figure(figsize=(10,6))
    sns.distplot(train[i][train[i] > 0].dropna(), kde = False, rug=False)
    plt.show()
for i in num_cols_log:
    test[i] = np.log(test[i] + 1)
    plt.figure(figsize=(10,6))
    sns.distplot(test[i][test[i] > 0].dropna(), kde = False, rug=False)
    plt.show()
plt.figure(figsize=(16,10))
sns.heatmap(train[num_cols].corr().abs(), vmin=0, vmax=1, annot=True)
num_cols
sns.boxplot(x=train.default, y=train.age)
sns.boxplot(x=train.default, y=train.days_passed)
sns.boxplot(x=train.default, y=train.decline_app_cnt)
sns.boxplot(x=train.default, y=train.score_bki)
sns.boxplot(x=train.default, y=train.region_rating)
sns.boxplot(x=train.default, y=train.income)
from scipy import stats

#train = train[(np.abs(stats.zscore(train)) < 3).all(axis = 1)]
#test = test[(np.abs(stats.zscore(test)) < 3).all(axis = 1)]
#train = train[((train.region_rating - train.region_rating.mean()) / train.region_rating.std()).abs() < 3]
def subset_by_iqr(df, column, whisker_width=1.5):
    q1 = df[column].quantile(0.25)                 
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR, including optional whiskers
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter]
#sns.boxplot(x=train.default, y=train.decline_app_cnt)
#train.decline_app_cnt = subset_by_iqr(train,'decline_app_cnt')
#test.decline_app_cnt = subset_by_iqr(test,'decline_app_cnt')
#sns.boxplot(x=train.default, y=train.decline_app_cnt)
sns.boxplot(x=train.default, y=train.score_bki)
train = subset_by_iqr(train,'score_bki',1)
test = subset_by_iqr(test,'score_bki',1)
sns.boxplot(x=train.default, y=train.score_bki)
sns.boxplot(x=train.default, y=train.region_rating)

train = subset_by_iqr(train,'region_rating')
test = subset_by_iqr(test,'region_rating')
sns.boxplot(x=train.default, y=train.region_rating)
sns.boxplot(x=train.default, y=train.income)
train = subset_by_iqr(train,'income')
test = subset_by_iqr(test,'income')
sns.boxplot(x=train.default, y=train.income)
#train = train[train.region_rating < train.region_rating.quantile(.75)]
#train = train[train.region_rating > train.region_rating.quantile(.25)]
label_encoder = LabelEncoder()

for column in bin_cols:
    train[column] = label_encoder.fit_transform(train[column])
    test[column] = label_encoder.fit_transform(test[column])
x_cat = OneHotEncoder(sparse = False).fit_transform(train[cat_cols].values)
y_cat = OneHotEncoder(sparse = False).fit_transform(test[cat_cols].values)

print(x_cat.shape)
print(y_cat.shape)
education_dummy = pd.get_dummies(train['education'])
education_dummy.head()
imp_num = Series(f_classif(train[num_cols], train['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh')
plt.title('Significance of num variables')
plt.xlabel('F-value')
# Значимость бинарных признаков

imp_bin = Series(mutual_info_classif(train[bin_cols], train['default'],
                                     discrete_features =True), index = bin_cols)
imp_bin.sort_values(inplace = True)
imp_bin.plot(kind = 'barh')
plt.title('Significance of bin variables')
# Значимость категориальных признаков

new_cat_cols = ['app_date_month', 'home_address', 'work_address', 'sna', 'first_time']

imp_cat = pd.Series(mutual_info_classif(pd.concat([train[new_cat_cols], education_dummy], axis=1),
                                        train['default'], discrete_features =True),
                    index = pd.concat([train[new_cat_cols], education_dummy], axis=1).columns)
imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh')
plt.title('Significance of cat variables')
def prepare_data():
    train = pd.read_csv('/kaggle/input/sf-dst-scoring/train.csv')
    test= pd.read_csv('/kaggle/input/sf-dst-scoring/test.csv')
    
    c_1 = collections.Counter(train['education'])
    c_2 = collections.Counter(test['education'])

    train['education'].fillna(c_1.most_common()[0][0], inplace=True)
    test['education'].fillna(c_2.most_common()[0][0], inplace=True)
    
    train.app_date = pd.to_datetime(train.app_date)
    test.app_date = pd.to_datetime(train.app_date)
    
    current_date = pd.to_datetime('28JUL2020')

    # Количество дней, прошедших со дня подачи заявки

    train['days_passed'] = (current_date - train.app_date).dt.days
    test['days_passed'] = (current_date - test.app_date).dt.days

    # Месяц подачи заявки

    train['app_date_month'] = train.app_date.dt.month
    test['app_date_month'] = test.app_date.dt.month
    
    train = subset_by_iqr(train, 'score_bki')
    test = subset_by_iqr(test, 'score_bki')  
    
    train = subset_by_iqr(train,'region_rating')
    test = subset_by_iqr(test,'region_rating')
    
    #train = subset_by_iqr(train,'income')
    #test = subset_by_iqr(test,'income')  
    
    return train, test
from sklearn.preprocessing import PolynomialFeatures

train, test = prepare_data()

label_encoder = LabelEncoder()
for column in bin_cols:
    train[column] = label_encoder.fit_transform(train[column])
    test[column] = label_encoder.fit_transform(test[column])   

x_cat = OneHotEncoder(sparse = False).fit_transform(train[cat_cols].values)
y_cat = OneHotEncoder(sparse = False).fit_transform(test[cat_cols].values)

poly = PolynomialFeatures(2)
x_tr = poly.fit_transform(train[num_cols].values)
y_test = poly.fit_transform(test[num_cols].values)

#Стандартизируем числовые переменные

x_num = StandardScaler().fit_transform(x_tr)
y_num = StandardScaler().fit_transform(y_test)

# Объединение

X = np.hstack([x_num, train[bin_cols].values, x_cat])
Y = train['default'].values

id_test = test.client_id
test = np.hstack([y_num, test[bin_cols].values, y_cat])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, shuffle = True)
from sklearn.model_selection import GridSearchCV

# Зададим ограничения для параметра регуляризации
C = np.logspace(0, 4, 10)

#methods = ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs']
methods = ['newton-cg']

penalty = ['l1', 'l2']
hyperparameters = dict(C=C, penalty=penalty)
model = LogisticRegression()
model.fit(X_train, y_train)

clf = GridSearchCV(model, hyperparameters, cv=4, verbose=0)

best_model = clf.fit(X_train, y_train)

best_penalty = best_model.best_estimator_.get_params()['penalty']
best_c = best_model.best_estimator_.get_params()['C']
best_solver = best_model.best_estimator_.get_params()['solver']
print('Лучший penalty:', best_penalty)
print('Лучшее C:', best_c)
print('Лучшее solver:', best_solver)

#Обучим модель
lgr = LogisticRegression(penalty = best_penalty, C=best_c, max_iter=700)
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

#log_error = logg_(y_test, probs)
#print('Logloss is', log_error)
    
lgr = LogisticRegression(penalty = best_penalty, C=best_c, solver=best_solver, max_iter=700)
lgr.fit(X, Y)
probs = lgr.predict_proba(test)
probs = probs[:,1]
my_submission = pd.DataFrame({'client_id': id_test, 'default': probs})
my_submission.to_csv('submission.csv', index=False)

my_submission