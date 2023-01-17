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
import numpy as np

import pandas as pd

import seaborn as sns

import missingno as msno

import matplotlib.pyplot as plt
file_path = "../input/pima-indians-diabetes-database/diabetes.csv"
data = pd.read_csv(file_path)
data.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
X = data.iloc[:, : -1]

X = (X - X.mean()) / X.std()

y = data["Outcome"]
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
t1 = LogisticRegression()

t1.fit(X_train, y_train)

t1.score(X_test, y_test)
p = t1.predict_proba(X_test)[:, 1]

roc_auc_score(y_test, p)