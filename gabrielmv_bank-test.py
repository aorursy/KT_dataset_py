from scipy.io.arff import loadarff

import pandas as pd

import matplotlib.pyplot as plt

import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
os.listdir('/kaggle/input/')
path_train = '../input/dados-bank/bankTR.arff'

path_teste = '../input/dados-bank/bankTE.arff'



row_data_train = loadarff(path_train)

train_df_data = pd.DataFrame(row_data_train[0])

row_data_test = loadarff(path_teste)

test_df_data = pd.DataFrame(row_data_test[0])



display(train_df_data.dtypes)

train_df_data.head()
train_df_data.describe()
train_df_data.hist(figsize=(10,8))
# Analisar a probabilidade de sobrevivência pelo Sexo

train_df_data[['age', 'duration']].groupby(['age']).mean()
fig, (axis1) = plt.subplots(1,1, figsize=(30,4))

sns.barplot(x = 'age', y = 'duration', data = train_df_data, ax=axis1)
columns = ['age', 'balance', 'day', 'duration']

pd.plotting.scatter_matrix(train_df_data[columns], figsize=(15, 10));
train_df_data.head()
def one_hot_encoding(df, column_name):

    df = pd.concat([df,pd.get_dummies(df[column_name], prefix=column_name)],axis=1)

    df.drop([column_name],axis=1, inplace=True)

    

    return df
columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']



for column in columns:

    train_df_data = one_hot_encoding(train_df_data, column)

    test_df_data  = one_hot_encoding(test_df_data, column)
test_df_data.head()
train_df_data = train_df_data.fillna(0)

test_df_data = test_df_data.fillna(0)
x_train = train_df_data.drop(['y'], axis = 1)
x_test = test_df_data.drop(['y'], axis = 1)
y_train = train_df_data[['y']]
y_test = test_df_data[['y']]
def true_or_false(label):

    return 1 if b'yes' in label else 0
for i, label in y_test.iterrows():

    y_test.iloc[i, 0] = true_or_false(label[0])

    

for i, label in y_train.iterrows():

    y_train.iloc[i, 0] = true_or_false(label[0])
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state=0)

rf = rf.fit(x_train, y_train)



print(f'model score: {rf.score(x_test, y_test) * 100}%')
y_pred = rf.predict(x_test)
confusion_matrix(y_test, y_pred)
print(f'model accuracy: {accuracy_score(y_test, y_pred) * 100}%')
print(f'kappa score: {cohen_kappa_score(y_test, y_pred) * 100}%')
clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)



print(f'model score: {rf.score(x_test, y_test) * 100}%')
tree.plot_tree(clf.fit(x_train, y_train)) 