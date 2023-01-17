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
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import seaborn as sns
import matplotlib.pyplot as plt

# Draw Decision Tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df_train = pd.read_csv('../input/titanic/train.csv')
df_train.head()
df_train.info()
df_train.shape
# ---> Let's look at the data statistics of df_train

df_train.describe()
df_train.describe(include='O')
# !!! We found that information in column Age, Cable, Embarked of df_train was missing and less than 891
# This is the answer for data_test

df_gen = pd.read_csv('../input/titanic/gender_submission.csv')
df_gen.head()
df_gen.shape
# Fill in with the median value in the age column.

df_trainNew = df_train.copy()
val = df_trainNew['Age'].median()
print('Fill in age of missing information = ',val)
df_trainNew['Age'].fillna(val, inplace = True)
df_trainNew.describe()
# Delete the columns in the DataFrame.

df_trainNew.drop(['PassengerId','Ticket','Name','Cabin'], axis=1, inplace=True)
df_trainNew.head()
# Using One-Hot Encoding to create dummy column
df_trainNew = pd.get_dummies(df_trainNew, columns=['Embarked','Sex'], dummy_na=True)
df_trainNew
# Using Min-Max Normalization to duplicate Age column and Fare column.

df_trainNew["Age"] = (df_trainNew["Age"] - df_trainNew["Age"].min()) / (df_trainNew["Age"].max() - df_trainNew["Age"].min())
df_trainNew["Fare"] = (df_trainNew["Fare"] - df_trainNew["Fare"].min()) / (df_trainNew["Fare"].max() - df_trainNew["Fare"].min())
df_trainNew
from sklearn.preprocessing import MinMaxScaler

X = df_trainNew.drop('Survived', axis=1)
y = df_trainNew['Survived']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = np.array(y)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from sklearn.metrics import confusion_matrix

dt = DecisionTreeClassifier()
nb = GaussianNB()
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

models_lst = [dt, nb, mlp]
from sklearn.model_selection import KFold

for model in models_lst:
    rec_lst = []
    pre_lst = []
    f1_lst = []
    name = model.__class__.__name__
    print(name)
    print('************************************')
    kf = KFold(n_splits=5, random_state=69, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        rec = recall_score(y_test, pred)
        pre = precision_score(y_test, pred)
        f1 = precision_score(y_test, pred)
        
        rec_lst.append(rec)
        pre_lst.append(pre)
        f1_lst.append(f1)
    print('Racall:', rec_lst)
    print('Precision:', pre_lst)
    print('AVG F-Measure:', np.array(f1_lst).mean())
    print('************************************')
    print('\n')