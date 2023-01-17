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

#Dataset.head(50)
#Dataset.describe()
#print(Dataset['state'].unique())
#欠損値の確認

#print(Dataset.isnull().sum(axis = 0))
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

df_state = [1 if i == 'successful' else 0 for i in df['state']]

df_state = pd.DataFrame(df_state, columns=['state'])

df = df.drop('state', axis=1)

df = pd.concat([df, df_state], axis=1)

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

onehot_main_category = pd.get_dummies(onehot['main_category'])

onehot_category = pd.get_dummies(onehot['category'])



df_dataset = pd.concat([data_values_std, onehot_currency, onehot_country, onehot_main_category, df['state']], axis=1)

df_dataset.head(50)
df_dataset.corr().style.background_gradient().format('{:.2f}')

#ほぼStateと相関がない
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(onehot_currency.values, df_dataset['state'].values, test_size=0.2, random_state=0)



from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier()

RandomForest.fit(X_train, Y_train)

print("RandomForest(Train) = ", round(RandomForest.score(X_train, Y_train) * 100, 2))

print("RandomForest(val) = ", round(RandomForest.score(X_val, Y_val) * 100, 2))



fti = RandomForest.feature_importances_   

print('Feature Importances:')

imp = pd.DataFrame(fti, index = pd.DataFrame(X_train).columns)

print(onehot_currency.columns, imp)

#5％以上はComics, Fashion, Food, Music, Technology, Theater→この通貨でなければ期間とゴール金額が支配的

onehot_currency_selected = onehot_currency[['AUD', 'CAD', 'EUR', 'GBP', 'USD']]
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(onehot_country.values, df_dataset['state'].values, test_size=0.2, random_state=0)



from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier()

RandomForest.fit(X_train, Y_train)

print("RandomForest(Train) = ", round(RandomForest.score(X_train, Y_train) * 100, 2))

print("RandomForest(val) = ", round(RandomForest.score(X_val, Y_val) * 100, 2))



fti = RandomForest.feature_importances_   

print('Feature Importances:')

imp = pd.DataFrame(fti, index = pd.DataFrame(X_train).columns)

print(onehot_country.columns, imp)

#5％以上はCA, GB, IT, NL, US→この国でなければ期間とゴール金額が支配的

onehot_country_selected = onehot_country[['CA', 'GB', 'IT', 'US']]
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(onehot_main_category.values, df_dataset['state'].values, test_size=0.2, random_state=0)



from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier()

RandomForest.fit(X_train, Y_train)

print("RandomForest(Train) = ", round(RandomForest.score(X_train, Y_train) * 100, 2))

print("RandomForest(val) = ", round(RandomForest.score(X_val, Y_val) * 100, 2))



fti = RandomForest.feature_importances_   

print('Feature Importances:')

imp = pd.DataFrame(fti, index = pd.DataFrame(X_train).columns)

print(onehot_main_category.columns, imp)

#5％以上はComics, Fashion, Food, Music, Technology, Theater→このmain_categoryでなければ期間とゴール金額が支配的

onehot_main_category_slected = onehot_main_category[['Comics', 'Fashion', 'Food', 'Music', 'Technology']]
df_dataset_selected = pd.concat([data_values_std, onehot_currency_selected, onehot_country_selected, onehot_main_category_slected, df['state']], axis=1)

df_dataset_selected.head(50)
X_data1 = df_dataset_selected.drop('state', axis=1)

X_data1.tail()
Y_data = df_dataset['state']

print(type(Y_data))

Y_data.tail()
X_data_val1 = X_data1.values

Y_data_val = Y_data.values
#Classification

from sklearn.model_selection import train_test_split,  GridSearchCV

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_data_val1, Y_data_val, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg = logreg.fit(X1_train, Y1_train)

print("LogisticRegression(Train) = ", round(logreg.score(X1_train, Y1_train) * 100, 2))

print("LogisticRegression(val) = ", round(logreg.score(X1_test, Y1_test) * 100, 2))



from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier()

RandomForest.fit(X1_train, Y1_train)

print("RandomForest(Train) = ", round(RandomForest.score(X1_train, Y1_train) * 100, 2))

print("RandomForest(val) = ", round(RandomForest.score(X1_test, Y1_test) * 100, 2))



"""

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X1_train, Y1_train)

print("Decision_tree(Train) = ", round(decision_tree.score(X1_train, Y1_train) * 100, 2))

print("Decision_tree(val) = ", round(decision_tree.score(X1_test, Y1_test) * 100, 2))

"""
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
X_data_val2 = df_dataset.drop('state', axis=1)

X_data_val2.tail()
#NeuralNetwork

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X_data_val2, Y_data_val, test_size=0.2, random_state=0)

parameters = {'hidden_layer_sizes':[(100,), (100, 100), (50, 100, 50)],  'solver':['adam', 'sgd'], 'activation':['tanh', 'relu', 'identity']} # ここを編集する

model = MLPClassifier(random_state=0, max_iter=100, early_stopping=True)

clf = GridSearchCV(model, parameters, cv=5)

clf.fit(X2_train, Y2_train)

print(clf.best_params_, clf.best_score_)
NN = MLPClassifier(hidden_layer_sizes=(100, 100), solver='adam', activation='tanh', random_state=0, max_iter=100, early_stopping=True)

NN.fit(X2_train, Y2_train)

print("Neural Network(Train) = ", round(NN.score(X2_train, Y2_train) * 100, 2))

print("Neural Network(val) = ", round(NN.score(X2_test, Y2_test) * 100, 2))
print('Neural Network')

Y2_pre = NN.predict(X2_test)#Logistic Regression

#1=positive

#0=negative

C_mat = pd.DataFrame(confusion_matrix(Y2_test, Y2_pre), 

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