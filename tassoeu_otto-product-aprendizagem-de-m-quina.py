# coding: utf-8



# para análise dos dados 

from matplotlib import pyplot as plt 

%matplotlib inline

import seaborn as sns



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import log_loss



print('Importando wrapper sklearn para xgboost... ', end='')

# Wrapper xgboost -> sklearn

import sys

import math



import numpy as np



sys.path.append('xgboost/wrapper/')

import xgboost as xgb
class XGBoostClassifier():

    def __init__(self, num_boost_round=10, **params):

        self.clf = None

        self.num_boost_round = num_boost_round

        self.params = params

        self.params.update({'objective': 'multi:softprob'})



    def fit(self, X, y, num_boost_round=None):

        num_boost_round = num_boost_round or self.num_boost_round

        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))

        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])

        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)



    def predict(self, X):

        num2label = dict((i, label)for label, i in self.label2num.items())

        Y = self.predict_proba(X)

        y = np.argmax(Y, axis=1)

        return np.array([num2label[i] for i in y])



    def predict_proba(self, X):

        dtest = xgb.DMatrix(X)

        return self.clf.predict(dtest)



    def score(self, X, y):

        Y = self.predict_proba(X)

        return 1 / logloss(y, Y)



    def get_params(self, deep=True):

        return self.params



    def set_params(self, **params):

        if 'num_boost_round' in params:

            self.num_boost_round = params.pop('num_boost_round')

        if 'objective' in params:

            del params['objective']

        self.params.update(params)

        return self

    

    

def logloss(y_true, Y_pred):

    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))

    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)

print('Concluido.')
# Leitura dos dados

print('Lendo dados de treinamento.. ', end='')

data = pd.read_csv("../input/train.csv")

data['id'] = data['id'].astype(str)



data_target = data['target']

data_features = data.drop(['target', 'id'], axis=1)

data_columns = data_features.columns

print('Concluido.')
order = sorted(set(train_df['target']))

sns.countplot(x='target', data=data,order=order)

plt.grid()

plt.title("Nº of Product of Each Class")

plt.figure(num=None, figsize=(20, 30), dpi=80, facecolor='w', edgecolor='k')
wt = data.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features")
# Normalização dos dados

print('Normalizando os dados de treinamento... ', end='')

scaler = StandardScaler()

scaler.fit(data_features, data_target)

scaled_data = scaler.transform(data_features)



data_features = pd.DataFrame(scaled_data, columns=data_columns)

print('Concluido.')



# Reconstruindo dados

print('Gerando nova tabela de dados de treinamento... ', end='')

data_rebuilt = np.column_stack((data_features, data_target))

data_columns_2 = np.asarray(data_columns.tolist() + ['target'])

data_columns_2

train_scaled_balanced = pd.DataFrame(data_rebuilt, columns=data_columns_2)

print('Concluido.')



# Dados de teste

print('Lendo dados de teste... ', end='')

test = pd.read_csv("../input/test.csv")



test_features = test.drop(['id'], axis=1)

test_id = test['id']

test_columns = test.columns

print('Concluido')



# Normalização dos dados

print('Normalizando os dados de teste... ', end='')

scaled_test_features = scaler.transform(test_features)

print('Concluido.')



# Reconstruindo dados

print('Gerando nova tabela de dados de teste... ', end='')

data_rebuilt = np.column_stack((test_id, scaled_test_features))

test_scaled = pd.DataFrame(data_rebuilt, columns=test_columns)

test_scaled ['id'] = test_scaled['id'].astype(int)

print('Concluido.')



# Gerando arquivos

train_scaled_balanced.to_csv('train_sb.csv', index=False)

test_scaled.to_csv('test_s.csv', index=False)
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
order = sorted(set(train_df['target']))

sns.countplot(x='target', data=train_df,order=order)

plt.grid()

plt.title("Nº of Product of Each Class")

plt.figure(num=None, figsize=(20, 30), dpi=80, facecolor='w', edgecolor='k')
cls1 = train_df[train_df.target=='Class_1']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_1")
cls1 = train_df[train_df.target=='Class_2']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_2")
cls1 = train_df[train_df.target=='Class_3']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_3")
cls1 = train_df[train_df.target=='Class_4']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_4")
cls1 = train_df[train_df.target=='Class_5']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_5")
cls1 = train_df[train_df.target=='Class_6']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_6")
cls1 = train_df[train_df.target=='Class_7']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_7")
cls1 = train_df[train_df.target=='Class_8']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_8")
cls1 = train_df[train_df.target=='Class_9']

wt = cls1.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features in Class_9")