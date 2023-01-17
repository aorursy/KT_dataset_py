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
# libs

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import *

from sklearn.ensemble import *

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

data
# insights about the data

data.describe()
x = data.loc[:,:"time"]

y = data.loc[:,["DEATH_EVENT"]]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)
clf_dt = DecisionTreeClassifier(criterion="gini",max_features="log2",max_depth=2,random_state=1)

clf_dt = clf_dt.fit(x_train,y_train)

clfdt_pred = clf_dt.predict(x_test)

clf_dt.score(x_test,y_test)
# building heatmap

cf_matrix = confusion_matrix(y_test, clfdt_pred)

sns.heatmap(cf_matrix, cmap='Oranges',annot = True)
x = data.loc[:,:"time"]

y = data.loc[:,["DEATH_EVENT"]]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)
clf_random = RandomForestClassifier(max_depth=4,max_features="log2",random_state=4)

clf_random = clf_random.fit(x_train,y_train)

clfrandom_pred = clf_random.predict(x_test)

clf_random.score(x_test,y_test)
# building heatmap

cf_matrix = confusion_matrix(y_test, clfrandom_pred)

sns.heatmap(cf_matrix, cmap='Oranges',annot = True)
x = data.loc[:,:"time"]

y = data.loc[:,["DEATH_EVENT"]]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)
clf_grad = GradientBoostingClassifier(learning_rate=0.1,max_depth=2)

clf_grad.fit(x_train,y_train)

clfgrad_pred = clf_grad.predict(x_test)

clf_grad.score(x_test,y_test)
# building heatmap

cf_matrix = confusion_matrix(y_test, clfgrad_pred)

sns.heatmap(cf_matrix, cmap='Oranges',annot = True)