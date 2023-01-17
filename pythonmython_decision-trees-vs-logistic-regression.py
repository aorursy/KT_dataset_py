# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

#Why? from sklearn.grid_search import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/add.csv', skipinitialspace=True, na_values='?')
df.head(12)
X = df.iloc[:,1:-1]
X = X.fillna(-1)
y = df.iloc[:,-1]
y = [1 if e == 'ad.' else 0 for e in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
models = [('LR',LogisticRegression()), ('clf', DecisionTreeClassifier(criterion= 'entropy', max_depth=150,min_samples_split=3))]

for name,model in models:
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    print('\n model: {}\'s classificaition report is \n\n {}'.format(name, classification_report(y_predict,y_test)))

