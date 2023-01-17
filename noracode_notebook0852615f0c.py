# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/tes')

print(df_train.columns.values)
y = df_train['SalePrice']
X = df_train.drop('SalePrice', axis=1)
features = ['OverallCond', 'YearBuilt']

dtrain = xgb.DMatrix(X[features].as_matrix(), label=y.as_matrix())
model = xgb.train({}, dtrain)
xgb.plot_importance(model)
model.predict(df_test[['']])