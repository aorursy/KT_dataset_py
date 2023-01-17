# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#generating data sets
train_data = train.copy()
test_data = test.copy()

vectors = test_data.columns

train_data = train_data.loc[:, vectors]
test_data = test_data.loc[:, vectors]
train_data.head()
test_data.head()
drop_list = []

for i in test_data.columns:
    if test_data[i].isna().sum() >= test_data[i].notnull().sum():
        drop_list.append(i)

for i in train_data.columns:
    if train_data[i].isna().sum() >= train_data[i].notnull().sum():
        drop_list.append(i)
        
print(drop_list)

drop_list = list(set(drop_list))

print(drop_list)
# dropping too void columns on both dataframes
train_data.drop(drop_list, axis=1, inplace=True)
test_data.drop(drop_list, axis=1, inplace=True)
for i in train_data.columns:
    nulls_value = train_data[i].isna().sum()
    message = "Column {} has {} nulls".format(i, nulls_value)
    print(message)
for i in test_data.columns:
    nulls_value = test_data[i].isna().sum()
    message = "Column {} has {} nulls".format(i, nulls_value)
    print(message)
# checking correlations

def plot_correlations(data):
    corr = data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()

plot_correlations(test_data.copy())
vector = plot_correlations(train_data.copy())
aux = train.copy()
aux2 = train.copy()

aux = aux.loc[:, test_data.columns]
aux['NU_NOTA_MT'] = aux2.NU_NOTA_MT

c = aux.corr()
c.NU_NOTA_MT.sort_values()
new_vector_training = [
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5',
    'NU_NOTA_COMP3',
    'NU_NOTA_REDACAO',
    'NU_NOTA_LC',
    'NU_NOTA_CH',
    'NU_NOTA_CN',
    'NU_NOTA_MT'
]

new_vector_test = [
    'NU_INSCRICAO',
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5',
    'NU_NOTA_COMP3',
    'NU_NOTA_REDACAO',
    'NU_NOTA_LC',
    'NU_NOTA_CH',
    'NU_NOTA_CN'
]

train_data = train.copy()
train_data = train_data.loc[:, new_vector_training]
train_data.dropna(subset=['NU_NOTA_MT'], inplace=True)
train_data.head()
y = train_data.NU_NOTA_MT
X = train_data.drop(['NU_NOTA_MT'], axis=1)

validation_data = test.copy()
validation_data_1 = validation_data.loc[:, new_vector_test]
validation_data_2 = validation_data.loc[:, new_vector_test]

train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state = 0)

model = XGBRegressor(n_estimators=200, learning_rate=0.1)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(validation_X, validation_y)], verbose=False)

validation_data_1.drop(['NU_INSCRICAO'], axis=1, inplace=True)
predicted_nota = model.predict(validation_data_1)
answer_df = pd.DataFrame({'NU_INSCRICAO': validation_data_2.NU_INSCRICAO, 'NU_NOTA_LC': validation_data_2.NU_NOTA_LC, 'NU_NOTA_MT_PREDICT': predicted_nota})
# almost there... now let's replace any note on math with None when note on LC is NaN (this means a missing value, once both are made on the same day

def replace_notes(row):
    if row.NU_NOTA_LC == np.NaN:
        return np.NaN
    return row.NU_NOTA_MT_PREDICT

answer_df['NU_NOTA_MT'] = answer_df.apply(replace_notes, axis='columns')
answer_df.loc[answer_df.NU_NOTA_LC.isna(), ['NU_NOTA_MT']] = np.NaN
answer_df_final = answer_df.loc[:, ['NU_INSCRICAO', 'NU_NOTA_MT']]

# answer_df.head()
# answer_df_final.head()
answer_df_final.to_csv('answer.csv', index=False)