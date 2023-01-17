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
import warnings

warnings.simplefilter('ignore')



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import random

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, make_scorer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC



from skopt import BayesSearchCV

# parameter ranges are specified by one of below

from skopt.space import Real, Categorical, Integer



#classifiers

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import GradientBoostingClassifier

from catboost import CatBoostClassifier, Pool

import xgboost as xgb

from lightgbm import LGBMClassifier

from sklearn.naive_bayes import MultinomialNB
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.head()
train_data.info()
def lower_casing(dataframe):

    dataframe.columns = dataframe.columns.str.lower()# перевод к нижнему регистру названий столбцов

    for i in range (len(dataframe.columns)):

        

        if dataframe.columns[i] != 'passengerid':# данные столбца customerid не изменяем

            try:

                dataframe[dataframe.columns[i]] = dataframe[dataframe.columns[i]].str.lower()

            except:

                dataframe[dataframe.columns[i]] = dataframe[dataframe.columns[i]]
lower_casing(train_data)

lower_casing(test_data)
def miss_inf(data, name):

    shape_data = data.shape

    miss_data = (data.isnull().sum()/data.shape[0]*100).sort_values(ascending = False).head(10)

    print()

    print(name)

    print()

    print('Количество строк, столбцов в ' + name + ':', shape_data)

    print()

    print('Доля пропущенных значений в столбцах (топ-10):')

    print(miss_data)

    print()

    print('Количество дубликатов в ' + name + ':', data.duplicated().sum())
miss_inf(train_data, 'train_data')
test_data.head()
test_data.info()
miss_inf(test_data, 'test_data')
sns.set_style('whitegrid')

plt.figure(figsize=(16, 5))

sns.distplot(train_data['age'][train_data['survived']== 1], bins=30, hist=False, label='Выжившие')

sns.distplot(train_data['age'][train_data['survived']== 0], bins=30, hist=False, label='Погибшие')

plt.title("Распределение погибших в зависимости от возраста")

plt.xlabel('Возраст')

plt.ylabel('Количество человек')

plt.legend()

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(16, 5))

sns.violinplot(x="survived", y="age", hue="sex", data=train_data, palette="coolwarm")

plt.title("Распределение погибших в зависимости от возраста и пола")

plt.legend()

plt.show()
surv_piv = train_data.pivot_table(index='sex', columns = 'survived', values = 'passengerid', aggfunc = 'count')

surv_piv['ratio'] = surv_piv[1] / (surv_piv[0] + surv_piv[1])

surv_piv['ratio']
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.barplot(x='sex', y='survived', data=train_data, palette="coolwarm")

plt.title("Количество погибших в зависимости от пола")

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(16, 5))

sns.distplot(train_data['age'], bins=30)

plt.title("Распределение пассажиров по возрасту")

plt.legend()

plt.show()
#тренировочная выборка

train_data.loc[train_data['age'] <= 16, 'f_age'] = 'child'

train_data.loc[(train_data['age'] > 16) & (train_data['age'] <= 25), 'f_age'] = 'young'

train_data.loc[(train_data['age'] > 25) & (train_data['age'] <= 45), 'f_age'] = 'adult'

train_data.loc[train_data['age'] > 45, 'f_age'] = 'senior'

#тестовая выборка

test_data.loc[test_data['age'] <= 16, 'f_age'] = 'child'

test_data.loc[(test_data['age'] > 16) & (test_data['age'] <= 25), 'f_age'] = 'young'

test_data.loc[(test_data['age'] > 25) & (test_data['age'] <= 45), 'f_age'] = 'adult'

test_data.loc[test_data['age'] > 45, 'f_age'] = 'senior'



train_data.head()
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.barplot(x='f_age', y='survived', hue='sex', data=train_data, palette="coolwarm")

plt.title("Количество погибших в зависимости от пола и возраста")

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.barplot(x='pclass', y='survived', hue='sex', data=train_data, palette="coolwarm")

plt.title("Количество погибших в зависимости от пола и класса билета")

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.distplot(train_data['fare'], kde = False, bins=100)

plt.title("Распределение пассажиров по стоимости билета")

plt.xlim(-1,100)

plt.legend()

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.boxplot(data=train_data, palette='rainbow', x="fare", orient='h')

plt.xlim(-1,100)

plt.legend()

plt.show()
train_data['fare'].describe()
#тренировочная выборка

train_data.loc[train_data['fare'] <= 15, 'f_fare'] = 'cheap'

train_data.loc[(train_data['fare'] > 15) & (train_data['fare'] <= 31), 'f_fare'] = 'middle'

train_data.loc[(train_data['fare'] > 31) & (train_data['fare'] <= 65), 'f_fare'] = 'expensive'

train_data.loc[train_data['fare'] > 65, 'f_fare'] = 'luxury'

#тестовая выборка

test_data.loc[test_data['fare'] <= 15, 'f_fare'] = 'cheap'

test_data.loc[(test_data['fare'] > 15) & (test_data['fare'] <= 31), 'f_fare'] = 'middle'

test_data.loc[(test_data['fare'] > 31) & (test_data['fare'] <= 65), 'f_fare'] = 'expensive'

test_data.loc[test_data['fare'] > 65, 'f_fare'] = 'luxury'



train_data.head()
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.barplot(x='f_fare', y='survived', hue='sex', data=train_data, palette="coolwarm")

plt.title("Количество погибших в зависимости от пола и класса билета")

plt.show()
train_data['fam_size'] = train_data['sibsp'] + train_data['parch'] + 1

test_data['fam_size'] = test_data['sibsp'] + test_data['parch'] + 1



train_data['fam_size'] = train_data['fam_size'].replace([5,6,7,8,11],1)

test_data['fam_size'] = test_data['fam_size'].replace([5,6,7,8,11],1)



#тренировочная выборка

train_data.loc[(train_data['sibsp'] == 0) & (train_data['parch'] == 0), 'is_family'] = 0

train_data.loc[(train_data['sibsp'] != 0) | (train_data['parch'] != 0), 'is_family'] = 1



#тестовая выборка



test_data.loc[(test_data['sibsp'] == 0) & (test_data['parch'] == 0), 'is_family'] = 0

test_data.loc[(test_data['sibsp'] != 0) | (test_data['parch'] != 0), 'is_family'] = 1





train_data.head()
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.barplot(x='is_family', y='survived', hue='sex', data=train_data, palette="coolwarm")

plt.title("Количество погибших в зависимости от пола и наличия семьи")

plt.show()
train_data.loc[(train_data['cabin'].isna() == False), 'is_cabin'] = 1

train_data.loc[(train_data['cabin'].isna() == True), 'is_cabin'] = 0



test_data.loc[(test_data['cabin'].isna() == False), 'is_cabin'] = 1

test_data.loc[(test_data['cabin'].isna() == True), 'is_cabin'] = 0
sns.set_style('whitegrid')

plt.figure(figsize=(12, 5))

sns.barplot(x='is_cabin', y='survived', data=train_data, palette="coolwarm")

plt.title("Количество погибших в зависимости от наличия названия номера")

plt.show()
def title(row):

    a = ['mrs','mr','miss','master']

    ret = []

    for b in a:

        if b in row['name']:

            ret.append(b)

    if len(ret)==0:

        if row['sex'] == 'male':

            ret = ['mr']

        else:

            ret = ['mrs']

    return ' '.join(ret)

train_data['title'] = train_data.apply(title, axis = 1)

test_data['title'] = test_data.apply(title, axis = 1)
#тренировочная выборка

train_data.loc[(train_data['title'] == 'mrs mr') | (train_data['title'] == 'mrs mr miss'), 'title'] = 'mrs'



#тестовая выборка

test_data.loc[(test_data['title'] == 'mrs mr') | (test_data['title'] == 'mrs mr miss'), 'title'] = 'mrs'
title_dict = dict(mrs = 0, miss = 1, master = 2, mr = 3)



train_data['title'] = train_data['title'].replace(title_dict)
test_data['title'] = test_data['title'].replace(title_dict)
train_data['embarked'] = train_data['embarked'].fillna('s')

test_data['fare'] = test_data.groupby(['pclass'])['fare'].transform(lambda x: x.fillna(x.median()))
avg_age_train = train_data.groupby(['sex', 'pclass'])['age'].median()

avg_age_test = test_data.groupby(['sex', 'pclass'])['age'].median()

avg_age_train
train_data['age'] = train_data.groupby(['sex', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))

test_data['age'] = test_data.groupby(['sex', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))



#округлим возраст младенцев

train_data.loc[train_data['age'] < 1, 'age'] = 1

test_data.loc[test_data['age'] < 1, 'age'] = 1
print(miss_inf(train_data, 'train_data'))

print(miss_inf(test_data, 'train_data'))
train_data
train_data.head()
train_data_drop = train_data.drop(['passengerid', 'name', 'age', 'sibsp', 'parch', 'ticket', 'cabin', 'f_age', 'f_fare'], axis=1)
test_data_drop = test_data.drop(['passengerid', 'name', 'age', 'sibsp', 'parch', 'ticket', 'cabin', 'f_age', 'f_fare'], axis=1)
train_data_drop.head()
# #тренировочная выборка

# pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# res = pf.fit_transform(train_data_drop[['pclass', 'title']]).astype('int')

# poly_features = pd.DataFrame(res, columns=['pclass^2', 'title^2', 'pc_t'])

# train_data_drop = train_data_drop.reset_index(drop=True)

# train_data_drop = train_data_drop.join(poly_features)

# train_data_drop['pclass^2'] = train_data_drop['pclass^2']**2

# train_data_drop['title^2'] = train_data_drop['title^2']**2
# # тестовая выборка

# pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# res = pf.fit_transform(test_data_drop[['pclass', 'title']]).astype('int')

# poly_features = pd.DataFrame(res, columns=['pclass^2', 'title^2', 'pc_t'])

# test_data_drop = test_data_drop.reset_index(drop=True)

# test_data_drop = test_data_drop.join(poly_features)

# test_data_drop['pclass^2'] = test_data_drop['pclass^2']**2

# test_data_drop['title^2'] = test_data_drop['title^2']**2
train_data_drop.head()
train_data_ohe = pd.get_dummies(train_data_drop, drop_first=True)

test_data_ohe = pd.get_dummies(test_data_drop, drop_first=True)

train_data_ohe.head()
X = train_data_ohe.drop('survived', axis=1)

y = train_data_ohe['survived']



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12345)



print('Размер обучающей выборки:', X_train.shape)

print('Размер валидационной выборки:', X_valid.shape)
X_train.head()
numeric = ['fare']



scaler = StandardScaler()

scaler.fit(X_train[numeric])



X_train[numeric] = scaler.transform(X_train[numeric])

X_valid[numeric] = scaler.transform(X_valid[numeric])

test_data_ohe[numeric] = scaler.transform(test_data_ohe[numeric])



X_train['fare'].describe().round(4)
train_data_ohe['survived'].value_counts(normalize=True)
# def upsample(features, target, repeat):

#     features_zeros = features[target == 0]

#     features_ones = features[target == 1]

#     target_zeros = target[target == 0]

#     target_ones = target[target == 1]



#     features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)

#     target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    

#     features_upsampled, target_upsampled = shuffle(

#         features_upsampled, target_upsampled, random_state=12345)

    

#     return features_upsampled, target_upsampled



# X_upsampled, y_upsampled = upsample(X_train, y_train, 2)
# print('Размеры таблицы features_upsampled:', X_upsampled.shape)

# print('Размеры таблицы target_upsampled:',y_upsampled.shape)

# print()

# print(y_upsampled.value_counts(normalize=True))
RND_ST = 42
def gridsearch(model, grid, features, target):



    CV = GridSearchCV(estimator = model, param_grid = grid, cv = 3, n_jobs=-1)

    CV.fit(features, target)

    

    print('accuracy_score = {:.2f}'.format(CV.best_score_))

    print(CV.best_params_)
def rand_search(model, grid, features, target):

    

    search = RandomizedSearchCV(model, grid, cv=7, n_jobs=-1)

    search.fit(features, target)

    

    print('accuracy_score = {:.2f}'.format(search.best_score_))

    print(search.best_params_)
def valid(model, fv, tv, name):

    

    predictions_valid = model.predict(fv)

   

    print(name + ' accuracy_score = {:.2f}'.format(accuracy_score(tv, predictions_valid)))
def final_pred(model, features, target, test):

   

    model.fit(features, target)

    

    pred_final = pd.DataFrame(model.predict(test), columns=['survived'])

 

    return pred_final
# def bayes_search(model, grid, features, target):

    

#     opt = BayesSearchCV(model, grid, n_iter=32)

#     opt.fit(features, target)

    

#     print('accuracy_score = {:.2f}'.format(opt.best_score_))

#     print(opt.best_params_)
#Логистическая ргерессия

LogisticRegression(random_state=RND_ST, n_jobs=-1)



#Случайный лес

rfc = RandomForestClassifier(random_state=RND_ST, n_jobs=-1)

rfc_params = dict(n_estimators=range(100, 600, 50), 

                  max_depth=range(3, 10, 2), 

                  max_features=range(3, 10, 2),

                  min_samples_split=[2,3,4]

                 )





#Градиентный бустинг

gbc = GradientBoostingClassifier(random_state=RND_ST)

gbc_params = dict(n_estimators=range(10, 300, 25),

                  max_depth=[3,4,7,10], 

                  learning_rate=[0.1,0.5,1],

                  max_leaf_nodes=range(1,6)

                 )

#CATBoost

cbc = CatBoostClassifier(random_state=RND_ST)

cbc_params = dict(n_estimators = range(50, 200, 25),

                  max_depth = [5, 7, 10],

                  learning_rate=[0.1,0.5,1],

                  verbose = [0]

                 )

#LGBM

lgb = LGBMClassifier(random_state=RND_ST)

lgb_params = dict(n_estimators = range(50, 200, 25),

                   max_depth = [5, 7, 10],

                  learning_rate=[0.1,0.5,1],

                   verbose = [0] 

                  )

# GaussianNB

gNB = GaussianNB()



# KNeighborsClassifier

knc = KNeighborsClassifier()

knc_params = dict(metric = ['manhattan', 'minkowski'],

                  n_neighbors = range(5,15),

                  leaf_size=range(30,50,2))

# SupportVector

svc = SVC(random_state=RND_ST)

svc_params = dict(

                  gamma = np.logspace(-6, -1, 5),

                  C = [0.1,1,10,100,1000],

                  tol=[1e-3, 1e-4, 1e-5])



# LinearSVC

lsv = LinearSVC(random_state=RND_ST)

lsv_params = dict(

                  C = [0.1, 1, 10, 100], 

                  penalty = ['l1', 'l2'],

                  max_iter = [1000,1500,2000])
lgr = LogisticRegression(random_state=RND_ST, n_jobs=-1)

cross_val_score(lgr, X_train, y_train, cv=7).mean()
gNB = GaussianNB()

cross_val_score(gNB, X_train, y_train, cv=7).mean()
gridsearch(rfc, rfc_params, X_train, y_train)
gridsearch(gbc, gbc_params, X_train, y_train)
gridsearch(cbc, cbc_params, X_train, y_train)
gridsearch(lgb, lgb_params, X_train, y_train)
gridsearch(knc, knc_params, X_train, y_train)
gridsearch(svc, svc_params, X_train, y_train)

gridsearch(lsv, lsv_params, X_train, y_train)
rfc_new = RandomForestClassifier(random_state=RND_ST, n_estimators = 200, max_features = 7, max_depth = 5, min_samples_split = 2)

cross_val_score(rfc_new, X_train, y_train, cv=7).mean()
gbc_new = GradientBoostingClassifier(random_state=RND_ST, n_estimators = 285, max_depth = 3, learning_rate = 0.1, max_leaf_nodes = 4)

cross_val_score(gbc_new, X_train, y_train, cv=7).mean()
cbc_new = CatBoostClassifier(random_state=RND_ST, n_estimators = 50, max_depth = 10, verbose = 0, learning_rate = 0.1 )

cross_val_score(cbc_new, X_train, y_train, cv=7).mean()
lgbm_new = LGBMClassifier(random_state=RND_ST, n_estimators = 75, max_depth = 7, verbose = 0, learning_rate = 0.1)

cross_val_score(lgbm_new, X_train, y_train, cv=7).mean()
knc_new = KNeighborsClassifier(leaf_size = 30, n_neighbors = 12, metric = 'manhattan')

cross_val_score(knc_new, X_train, y_train, cv=7).mean()
svc_new = SVC(random_state=RND_ST, C = 100, gamma = 0.005623413251903491, tol = 0.001)

cross_val_score(svc_new, X_train, y_train, scoring = 'accuracy', cv=7).mean()
lsv_new = LinearSVC(random_state=RND_ST, C = 10, max_iter = 1500, penalty = 'l2')

cross_val_score(lsv_new, X_train, y_train, scoring = 'accuracy', cv=7).mean()
vc = VotingClassifier([('clf1', gbc_new), ('clf2', rfc_new)], voting='soft')

cross_val_score(vc, X_train, y_train, cv=7).mean()
vc = VotingClassifier([('clf1', gbc_new), ('clf2', rfc_new)], voting='soft')

vc.fit(X_train, y_train)

valid(vc, X_valid, y_valid, 'Результат на валидационной выборке:')
vc_params = {'voting':['hard', 'soft'],

            'weights':[(1,1),(1,2),(2,1)]}
gridsearch(vc, vc_params, X, y)
vc_new = VotingClassifier([('clf1', gbc_new), ('clf2', rfc_new)], voting='hard', weights = (2, 1))

vc_new.fit(X_train, y_train)

valid(vc_new, X_valid, y_valid, 'Результат на валидационной выборке:')
## Подберем гиперпараметры для наилучшей модели
# gridsearch(gbc, gbc_params, X_upsampled, y_upsampled)
# get a stacking ensemble of models

def get_stacking():

    # define the base models

    level0 = list()

    level0.append(('lgr', LogisticRegression()))

    level0.append(('knc_new', KNeighborsClassifier(leaf_size = 30, n_neighbors = 9)))

    level0.append(('rfc_new', RandomForestClassifier(random_state=RND_ST, n_estimators = 125, max_features = 5, max_depth = 5)))

    level0.append(('gbc_new', GradientBoostingClassifier(random_state=RND_ST, n_estimators = 60, max_depth = 3, learning_rate = 0.1)))

    level0.append(('svc_new', SVC(random_state=RND_ST, C = 10, gamma = 0.005623413251903491, tol = 0.001)))



    # define meta learner model

    level1 = CatBoostClassifier(random_state=RND_ST, n_estimators = 50, max_depth = 10, verbose = 0, learning_rate = 0.1)

    # define the stacking ensemble

    scl = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

    return scl
# get a list of models to evaluate

def get_models():

    models = dict()

    models['lgr'] = LogisticRegression()

    models['knc_new'] = KNeighborsClassifier(leaf_size = 20, n_neighbors = 7)

    models['rfc_new'] = RandomForestClassifier(random_state=RND_ST, n_estimators = 125, max_features = 5, max_depth = 5)

    models['gbc_new'] = GradientBoostingClassifier(random_state=RND_ST, n_estimators = 60, max_depth = 3, learning_rate = 0.1)

    models['svc_new'] = SVC(random_state=RND_ST, C = 10, gamma = 0.005623413251903491, tol = 0.001)

    models['stacking'] = get_stacking()

    return models
# evaluate a give model using cross-validation

def evaluate_model(model, X, y):

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    return scores
# get the models to evaluate

models = get_models()
# evaluate the models and store results

results, names = list(), list()

for name, model in models.items():

    scores = evaluate_model(model, X_train, y_train)

    results.append(scores)

    names.append(name)

    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# importances = svc_new1.feature_importances_

# print("Feature importances:")

# for i in range(len(X_train.columns)):

#     print("{:2d}. feature '{:5s}' ({:.4f})".format(i+1, X_train.columns[i], importances[i]))
gbc_new1 = GradientBoostingClassifier(random_state=RND_ST, n_estimators = 285, max_depth = 3, learning_rate = 0.1, max_leaf_nodes = 4)

gbc_new1.fit(X_train, y_train)

valid(gbc_new1, X_valid, y_valid, 'Результат на валидационной выборке:')
pred_final = final_pred(gbc_new1, X_train, y_train, test_data_ohe)

submission = pd.DataFrame(test_data['passengerid'])

submission = submission.join(pred_final)

submission.to_csv('/kaggle/working/titanic_gradboost9.csv', index=False)

submission.head()