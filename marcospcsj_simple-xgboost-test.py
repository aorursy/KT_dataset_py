# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_treino = pd.read_csv('../input/train.csv')

df_treino.head()
df_teste =pd.read_csv('../input/test.csv')

df_teste.head()
train_columns = df_treino.drop(['ID_code','target'],axis =1).columns

train_columns
import xgboost as xgb

d_treino = xgb.DMatrix(df_treino[train_columns],label = df_treino['target'])

d_teste = xgb.DMatrix(df_teste[train_columns])
param_grid = {'max_depth':3,

                   'silent':1,

                   'eta':0.28071497637474263, # learning rate

                   'gamma':0,

                   'min_child_weight':0.2784483175645849,

                   'objective':'binary:logistic'

                  }
xgbmodel = xgb.train(param_grid, d_treino,500)

preds = xgbmodel.predict(d_teste)
dfprediction = df_teste[['ID_code']]

dfprediction['target']=preds
dfprediction.head()
# saving as regression

dfprediction.to_csv("submission23.csv", index=False)