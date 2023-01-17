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
import pandas as pd

import numpy as np

glass_dataset = pd.read_csv("../input/glass/glass.csv")
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

glass_dataset.info()

glass_dataset["Type"]

glass_dataset.iloc[:,9]
glass_dataset.iloc[:,:9]

train_set, test_set = train_test_split(glass_dataset.as_matrix(), test_size = 0.1, random_state = 42)

Y_train = train_set[:,9]

X_train = train_set[:,:9]

X_test = test_set[:,:9]

Y_test = test_set[:,9]

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators" : [4,6,8],"max_features" : [4,6,8]}

grid_search = GridSearchCV(rnd_clf, param_grid, cv = 3, scoring = 'accuracy')

grid_search.fit(X_train,Y_train)

grid_search.best_params_





from sklearn.metrics import accuracy_score

grid_search.predict(X_test)

Y_predicted = grid_search.predict(X_test).copy()

accuracy_score(Y_test,Y_predicted)
n = 0

m = 0

for a, b in Y_predicted, Y_test:

    m += 1

    if a == b:

        n += 1

n/m

   