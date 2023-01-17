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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Perceptron, RidgeCV, Lasso

from sklearn.decomposition import PCA



import torch

from torch import optim

from torch import nn

from torch.utils.data import Dataset, DataLoader

train_X = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_y = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_y_ns = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

test_X = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
train_X.head(2)
X_columns, y_columns = train_X.columns, train_y.columns

X_columns
sns.countplot(train_X['cp_type'])
sns.countplot(train_X['cp_dose'])
sns.countplot(train_X['cp_time'])
print("Train features shape:", train_X.shape, "Train Scored Targets:", train_y.shape)
def drop_cp(df):

    return df[df['cp_type'] == 'trt_cp'].reset_index(drop=True)

ind_tr = train_X[train_X['cp_type'] == 'ctl_vehicle'].index

ind_te = test_X[test_X['cp_type'] == 'ctl_vehicle'].index
#most_popular_occurancies

scored = train_y.sum()[1:].sort_values()

scored[:30].plot(kind='barh', fontsize=14, figsize=(5,20))
def normal_split(train_X, train_y):

    X_n_columns = len(train_X.columns)

    y_n_columns = len(train_y.columns)

    

    train_X_y = pd.merge(train_X, train_y ,on='sig_id')

    train_X_y.drop('sig_id', axis=1)

    train_X = train_X_y.iloc[:, 1:X_n_columns]

    train_y = train_X_y.iloc[:, X_n_columns:]

    return train_X, train_y



train_X, train_y = normal_split(train_X, train_y)

def preprocess(df):

    df = df.drop('sig_id', axis=1, errors='ignore')

    df = df.drop('cp_type', axis=1, errors='ignore')

    

    df['cp_time'] = df['cp_time'].map({24:1, 48:2, 72:3})

    df['cp_dose'] = df['cp_dose'].map({'D1':0, 'D2':1})

    

    g_features = [col for col in df.columns if col.startswith('g-')]

    c_features = [col for col in df.columns if col.startswith('c-')]

    

    pca_transformer_g = PCA(n_components=len(g_features) // 4)

    pca_transformer_c = PCA(n_components=len(c_features) // 4)

    

    pca_features_g = pd.DataFrame(pca_transformer_g.fit_transform(df[g_features]))

    pca_features_c = pd.DataFrame(pca_transformer_c.fit_transform(df[c_features]))



    new_df = pd.concat([df['cp_time'], df['cp_dose'], pca_features_g, pca_features_c], axis=1)

    

    

    '''

    for cell_type in ['c','g']:

        if cell_type == 'c':

            features = c_features

        else:

            features = g_features

                

        for time in [1, 2, 3]: 

            df['mean_{}_time{}'.format(cell_type, time)] = df.loc[df['cp_time']==time, features].mean(axis=1)

        df['mean_{}_dose0'] = df.loc[df['cp_dose']==0, features].mean(axis=1)

    df.fillna(0, inplace=True)

    '''

    return new_df



g_features = [col for col in train_X.columns if col.startswith('g-')]

c_features = [col for col in train_X.columns if col.startswith('c-')]



print(train_X.shape)

for  i in range(0, len(c_features), len(c_features)//10):

    sns.kdeplot(train_X.loc[:, c_features[i]])

    plt.show()
print(train_X.shape)

for  i in range(0, len(g_features), len(g_features)//10):

    sns.kdeplot(train_X.loc[:, g_features[i]])

    plt.show()
test_ids = test_X.sig_id

train_X, test_X = preprocess(train_X), preprocess(test_X)
#model = LinearRegression()

#model = DecisionTreeRegressor()

#model = Perceptron()

#model = RidgeCV()

model = Lasso()
model.fit(train_X.to_numpy(), train_y.to_numpy())
preds = model.predict(test_X)

print(preds.shape)

print(train_X.shape)
preds = pd.DataFrame(preds, columns=train_y.columns)

preds['sig_id'] = test_ids

cols = preds.columns.tolist()

cols = cols[-1:] + cols[:-1]

preds = preds[cols]
preds.set_index(keys='sig_id', drop=False)



preds.to_csv("submission.csv", index=None)