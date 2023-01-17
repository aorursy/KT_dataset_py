# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import keras

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
macro = pd.read_csv('../input/macro.csv')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
corr = macro.corr()
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Draw the heatmap using seaborn

sns.heatmap(corr, vmax=.8, square=True)
corr = train.corr()

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Draw the heatmap using seaborn

sns.heatmap(corr, vmax=.8, square=True)
y_train = train['price_doc']

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)

id_test = test.id
from sklearn import preprocessing

for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values)) 

        x_train[c] = lbl.transform(list(x_train[c].values))

        x_train.drop(c,axis=1,inplace=True)

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values)) 

        x_test[c] = lbl.transform(list(x_test[c].values))

        x_test.drop(c,axis=1,inplace=True) 

        

x_train = x_train.fillna(x_train.median())

x_test = x_test.fillna(x_test.median())
import xgboost as xgb

model = xgb.XGBRegressor()

model.fit(x_train, y_train)

print("Fitting done")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 30))

xgb.plot_importance(model, ax=ax)

print("Features importance done")
results = model.predict(x_test)
output = pd.DataFrame({'id': id_test, 'price_doc': results})

output.to_csv('predict.csv', index=False)