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
df_train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

df_test.head()
list(df_train.columns.values)
df_train.describe()
df_train[['bin_0']].describe()
df_train[['id']]
# designate target variable name

targetName = 'target'

targetSeries = df_train[targetName]

#remove target from current location and insert in collum 0

del df_train[targetName]

df_train.insert(0, targetName, targetSeries)

#reprint dataframe and see target is in position 0

df_train.head(10)
df_train_processed = pd.get_dummies(df_train)
df_train_processed.head()
#Add packages

#These are my standard packages I load for almost every project

%matplotlib inline 

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#From Scikit Learn

from sklearn import preprocessing

from sklearn.model_selection  import train_test_split, cross_val_score, KFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

#Notice I did not load my Datamining packages yet (ie decision tree). I will do that as I use them.
target = df_train_processed['target'].to_frame()
df_train_processed.iloc[:,2:16441]
#Decision Tree train model. Call up my model and name it clf

from sklearn import tree 

clf_dt = tree.DecisionTreeClassifier()

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_dt)

#Fit clf to the training data

clf_dt = clf_dt.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_dt = clf_dt.predict(features_test)