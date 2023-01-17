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
df_train = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/train.csv',index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/test.csv',index_col=0)
df_train["type"] = df_train["type"].map({'ependymoma':1,'glioblastoma':2,'medulloblastoma':3,'pilocytic_astrocytoma':4,'normal':0})
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



sns.countplot(df_train["type"])

plt.show()
X = df_train.drop(["type"], axis=1).values

y = df_train["type"].values

X_test = df_test.values
from sklearn.tree import DecisionTreeClassifier

"""from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 25)

params = {'max_depth':[25,30,35,40,45]}

gscv = GridSearchCV(clf, params, cv=5, scoring='f1_micro')

gscv.fit(X, y)"""
"""gscv.best_params_"""
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 25)
clf.fit(X,y)
clf.score(X,y)
p = clf.predict(X_test)
p
df_submit = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/sampleSubmission.csv',index_col=0)

df_submit["type"] = p

df_submit.to_csv("my_submission2")