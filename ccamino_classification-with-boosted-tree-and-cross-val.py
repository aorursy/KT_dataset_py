# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier

from xgboost import plot_importance

from xgboost import plot_tree

import matplotlib.pyplot as plt

from sklearn.cross_validation import  train_test_split

from matplotlib import pyplot

from sklearn import metrics

from sklearn.cross_validation import KFold, cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# loading the data

dataset =  pd.read_csv('../input/data.csv', header=0)
dataset.head()
dataset = dataset.drop("id",1)

dataset = dataset.drop("Unnamed: 32",1)

d = {'M' : 0, 'B' : 1}

dataset['diagnosis'] = dataset['diagnosis'].map(d)
dataset.head()
features = list(dataset.columns[1:31])
model = XGBClassifier()

model.fit(dataset[features],dataset['diagnosis'])

# plot feature importance

plot_importance(model)

pyplot.show()
X_train, X_test, y_train, y_test = train_test_split( dataset[features], dataset['diagnosis'], test_size=0.30, random_state=42)

model.fit(X_train,y_train)

predictions = model.predict(X_test)     

metrics.accuracy_score(y_test, predictions)
Kfold = KFold(len(dataset),n_folds=10,shuffle=False)
Kfold = KFold(len(dataset),n_folds=10,shuffle=False)

print("KfoldCrossVal score using BoostedTree is %s" %cross_val_score(model,dataset[features],dataset['diagnosis'],cv=10).mean())