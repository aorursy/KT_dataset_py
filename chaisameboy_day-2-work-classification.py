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
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#データのロード

Dataset = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")



#データの表示

Dataset.head(50)
#データの大きさ

print(Dataset.shape)

#columns

print(Dataset.columns)

print(Dataset.dtypes)
Dataset.describe()
print(Dataset['state'].unique())
#欠損値の確認

print(Dataset.isnull().sum(axis = 0))
#Successとfailedだけを使用

df_fialed = Dataset[(Dataset['state'] == 'failed')]

df_successful = Dataset[Dataset['state'] == 'successful']
plt.figure(figsize = (15, 10))

scatter = plt.bar([1, 2], [len(df_fialed), len(df_successful)], width=0.5, color=['blue', 'red'], tick_label=['Failed', 'Successful'])

#scatter.set(ylim = (0, 200))

plt.show()
#failedとSuccessfulを同数に

df_fialed = df_fialed.sample(n = len(df_successful), random_state = 0)

plt.figure(figsize = (15, 10))

scatter = plt.bar([1, 2], [len(df_fialed), len(df_successful)], width=0.5, color=['blue', 'red'], tick_label=['Failed', 'Successful'])

#scatter.set(ylim = (0, 200))

plt.show()

Dataset = pd.concat([df_fialed, df_successful], axis=0)

Dataset = Dataset.reset_index()

print(Dataset['state'].unique())

print(Dataset.shape)

print(Dataset.isnull().sum())
#使用するデータを選定

df = Dataset[['category', 'main_category', 'currency', 'country', 'deadline', 'launched', 'goal', 'usd_goal_real', 'state' ]]

df['state'] = [1 if i == 'successful' else 0 for i in df['state']]

df.tail(10)
#standizaton_process

import copy

#period

import datetime

datetime = copy.deepcopy(df)

datetime_launched = pd.to_datetime(datetime['launched'], format='%Y-%m-%d %H:%M:%S')

datetime_deadline = pd.to_datetime(datetime['deadline'], format='%Y-%m-%d %H:%M:%S')

period = datetime_deadline - datetime_launched

period_df = pd.DataFrame(period.values / np.timedelta64(1, 's'), columns=['period'])



#usd_goal_real

goal = copy.deepcopy(df)

goal_df = pd.DataFrame(goal['usd_goal_real'], columns=['usd_goal_real'])



#standizaton

data_values = pd.concat([period_df, goal_df,], axis=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

data_values_std = pd.DataFrame(sc.fit_transform(data_values), columns=['period_std','usd_goal_real_std'])



#onehot encording



onehot = copy.deepcopy(df)

onehot_currency = pd.get_dummies(onehot['currency'])

onehot_country = pd.get_dummies(onehot['country'])

onehot_category = pd.get_dummies(onehot['main_category'])





df_dataset = pd.concat([data_values_std, onehot_currency, onehot_country, onehot_category, df['state']], axis=1)

df_dataset.head(50)
df_dataset.corr().style.background_gradient().format('{:.2f}')

#ほぼStateと相関がない
#stepwise method

from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()

rfecv = RFECV(estimator, cv=54, scoring='accuracy')

Y_data = df_dataset['state']

X_data = df_dataset.drop('state', axis=1)

Y = Y_data.values

X = X_data.values



# fitで特徴選択を実行

rfecv.fit(X, Y)

# 特徴のランキングを表示（1が最も重要な特徴）

print('Feature ranking: \n{}'.format(rfecv.ranking_))
#datasetの変数を選定

remove_idx = ~rfecv.support_

remove_feature = X_data.columns[remove_idx]

print(remove_feature)
#重要度の低い特徴を削除

X_data = df_dataset.drop(['state','DKK', 'JPY', 'SGD', 'AU', 'DK', 'ES', 'JP', 'N,0"', 'NL'], axis=1)

X_data.head()
Y_data = df_dataset['state']

print(type(Y_data))

Y_data.tail()
X_data_val = X_data.values

Y_data_val = Y_data.values
#Classification

from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_data_val, Y_data_val, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg = logreg.fit(X1_train, Y1_train)

print("LogisticRegression(Train) = ", round(logreg.score(X1_train, Y1_train) * 100, 2))

print("LogisticRegression(val) = ", round(logreg.score(X1_test, Y1_test) * 100, 2))





from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X1_train, Y1_train)

print("Decision_tree(Train) = ", round(decision_tree.score(X1_train, Y1_train) * 100, 2))

print("Decision_tree(val) = ", round(decision_tree.score(X1_test, Y1_test) * 100, 2))



from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier()

RandomForest.fit(X1_train, Y1_train)

print("RandomForest(Train) = ", round(decision_tree.score(X1_train, Y1_train) * 100, 2))

print("RandomForest(val) = ", round(decision_tree.score(X1_test, Y1_test) * 100, 2))
from sklearn.metrics import confusion_matrix

print('Logistic Regression')

Y1_pre = logreg.predict(X1_test)#Logistic Regression

#1=positive

#0=negative

C_mat = pd.DataFrame(confusion_matrix(Y1_test, Y1_pre), 

                        index=['1', '0'], 

                        columns=['1', '0'])

print(C_mat)

#正解率

accuracy = (C_mat['1'][0]+C_mat['0'][1])/(C_mat['0'][0]+C_mat['0'][1]+C_mat['1'][0]+C_mat['1'][1])

print('Accuracy = ', accuracy)



#適合率

precision = C_mat['1'][0] / (C_mat['1'][0] + C_mat['1'][1]) 

print('Precision  = ', precision)



#再現率

recall = C_mat['1'][1]/(C_mat['1'][0]+C_mat['0'][0])

print('Recall = ', recall)