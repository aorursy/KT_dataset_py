# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from random import randint, shuffle

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train_data/train_data.csv', sep=',')

df_test = pd.read_csv('../input/test_data/test_data.csv', sep=',')
list_engine_no = list(df_train['engine_no'].drop_duplicates())



ratio = .3

shuffle(list_engine_no)

engine_no_test = list_engine_no[:int(len(list_engine_no) * ratio)]

engine_no_train = [x for x in list_engine_no if x not in engine_no_test]
nan_column = df_train.columns[df_train.isna().any()].tolist()

const_columns = [c for c in df_train.columns if len(df_train[c].drop_duplicates()) <= 2]

print('Columns with all nan: \n' + str(nan_column) + '\n')

print('Columns with all const values: \n' + str(const_columns) + '\n')
metadata_columns = ['engine_no', 'time_in_cycles']

selected_features = [x for x in df_test.columns if x not in metadata_columns + nan_column + const_columns]
df_train_train = df_train[df_train['engine_no'].isin(engine_no_train)]

data_eval = df_train[df_train['engine_no'].isin(engine_no_test)]



X_train_train, y_train_train = df_train_train[selected_features], df_train_train['RUL'] 

X_eval, y_eval = data_eval[selected_features], data_eval['RUL']



X_train_all, y_train_all = df_train[selected_features], df_train['RUL']



X_test = df_test[selected_features]

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=20)

tree.fit(X_train_train, y_train_train)
print("Score on train data : " + str(tree.score(X_train_train, y_train_train)))

print("Score on eval data : " + str(tree.score(X_eval, y_eval)))
tree.fit(X_train_all, y_train_all)
df_train['pred_tree'] = tree.predict(X_train_all)

df_test['pred_tree'] = tree.predict(X_test)
df_train = df_train.sort_values(['engine_no', 'time_in_cycles'])

df_test = df_test.sort_values(['engine_no', 'time_in_cycles'])



# On prend la dernière prédiction du RUL

df_result = df_test.groupby('engine_no').last().reset_index()[['engine_no', 'pred_tree']]



# On convertit en binaire (RUL > 100 ?)

df_result['result'] = df_result['pred_tree'].map(lambda x: 0 if x > 100 else 1)

df_plot = df_train.copy()

df_plot = df_plot.sort_values(['engine_no', 'time_in_cycles'])

g = sns.PairGrid(data=df_plot, x_vars="RUL", y_vars=['RUL', 'pred_tree'], hue="engine_no", height=6, aspect=6,)

g = g.map(plt.plot, alpha=0.5)

g = g.set(xlim=(df_plot['RUL'].max(),df_plot['RUL'].min()))
df_result[['engine_no', 'result']].to_csv('submission.csv', index=False)