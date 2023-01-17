import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import json

from sklearn import preprocessing

from tqdm import tqdm

import random



# для детерминированности результатов

random_state = 777

np.random.seed(random_state)
train_data = pd.read_csv('../input/mlbootcampdigital2020task3/train.csv',

                         names=['order_date', 'requester', 'service', 'cpgu_user', 'service_title',

                               'receipt_mfc', 'order_number', 'mfc', 'internal_status', 'external_status',

                               'sub_department', 'creation_mfc', 'order_type', 'department_id',

                               'deleted', 'deleter_fk', 'custom_service_id',

                               'close_date', 'service_level', 'issue_date', 'change_timestamp'], skiprows=1)

train_data["label"] = np.nan
test_data = pd.read_csv('../input/mlbootcampdigital2020task3/test_ids.txt', sep=' ', header=None, names=['requester'], skiprows=1)
# nan заполняем -1

train_data.fillna(-1, inplace=True)
# удаляем последнюю строку

train_data.drop(train_data.tail(1).index,inplace=True)
# создаем финальный тренировочный датасет

#df = train_data.head(0)
#sampled_train_data = train_data.sample(n=400000, random_state=random_state)

'''

requesters = np.random.choice(list(train_data['requester'].unique()), size=60000, replace=False)

for requester in tqdm(requesters, position=0):

    sub_data = train_data[train_data['requester'] == requester]

    # создаем пары значений x,y 

    for i in range(len(sub_data)-1):

        new_row = sub_data.iloc[i]

        new_row['label'] = int(sub_data.iloc[i+1]['service_title'])

        df = df.append(new_row, ignore_index=False)

df["label"] = df["label"].astype(int)

'''
#df.to_csv("df.csv", index=False)

#df = pd.read_csv("../input/mfc-df2020/df.csv")
train_data['label'] = train_data.shift(periods=-1)['service_title']
train_data.head()
# Сформируем тестовый датасет

''' 

df_test = df.head(0)



for requester in tqdm(test_data['requester'], position=0):

    sub_data = train_data[train_data['requester'] == requester]

    if len(sub_data) > 0:

        df_test.loc[len(df_test)] = sub_data.iloc[len(sub_data) - 1]

    else:

        sub_data = train_data[train_data['requester'] == test_data['requester'][0]]

        sub_data['requester'] = requester

        df_test.loc[len(df_test)] = sub_data.iloc[len(sub_data) - 1]

'''
#df_test.to_csv("df_test.csv", index=False)

df_test = pd.read_csv("../input/mfc2020test/df_test.csv")
df_test.head(2)
# удаляем колонки с датой

train_data.drop(['order_date', 'close_date', 'issue_date', 'change_timestamp'], axis=1, inplace=True)
df_test.drop(['order_date', 'close_date', 'issue_date', 'change_timestamp'], axis=1, inplace=True)
# и остальные несущественные данные

train_data.drop(['order_number', 'deleter_fk', 'custom_service_id'], axis=1, inplace=True)
df_test.drop(['order_number', 'deleter_fk', 'custom_service_id'], axis=1, inplace=True)
# применяем label-encoding к данным

le_order = preprocessing.LabelEncoder()

le_order = le_order.fit(train_data['order_type'].apply(str))

train_data['order_type'] = le_order.transform(train_data['order_type'].apply(str))
le_service = preprocessing.LabelEncoder()

le_service = le_service.fit(train_data['service_level'].apply(str))

train_data['service_level'] = le_service.transform(train_data['service_level'].apply(str))
le_receipt_mfc = preprocessing.LabelEncoder()

le_receipt_mfc = le_receipt_mfc.fit(train_data['receipt_mfc'].apply(str))

train_data['receipt_mfc'] = le_receipt_mfc.transform(train_data['receipt_mfc'].apply(str))
le_mfc = preprocessing.LabelEncoder()

le_mfc = le_mfc.fit(train_data['mfc'].apply(str))

train_data['mfc'] = le_mfc.transform(train_data['mfc'].apply(str))
le_deleted = preprocessing.LabelEncoder()

le_deleted = le_deleted.fit(train_data['deleted'].apply(str))

train_data['deleted'] = le_deleted.transform(train_data['deleted'].apply(str))
# применяем label-encoding к данным

df_test['order_type'] = le_order.transform(df_test['order_type'].apply(str))
df_test['service_level'] = le_service.transform(df_test['service_level'].apply(str))
df_test['receipt_mfc'] = le_receipt_mfc.transform(df_test['receipt_mfc'].apply(str))
df_test['mfc'] = le_mfc.transform(df_test['mfc'].apply(str))
df_test['deleted'] = le_deleted.transform(df_test['deleted'].apply(str))
train_data.fillna(-100, inplace=True)



# извлекаем X/y

X_train = train_data.drop(['label', 'requester'], axis=1)

y_train = train_data['label']



# скалируем данные

scaler = preprocessing.MinMaxScaler()

scaler = scaler.fit(X_train)

X_train = scaler.transform(X_train)
# скалируем данные

X_test = scaler.transform(df_test.drop(['label', 'requester'], axis=1))
del df_test
del train_data
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(max_iter=1000, tol=1e-5, random_state=random_state)
clf.fit(X_train, y_train)
# получаем предсказания

test_data['service_title'] = clf.predict(X_test)
# сохраняем результаты модели

test_data.to_csv("submission.csv", index=False)