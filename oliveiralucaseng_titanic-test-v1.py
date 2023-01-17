# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
result = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.head()
#Binary Encoding for sex -> male = 1, female = 0
df_test.Sex = df_test.Sex.map({'male': 1, 'female': 0})
#Binary Encoding for sex -> male = 1, female = 0
df_train.Sex = df_train.Sex.map({'male': 1, 'female': 0})
df_train.describe()
df_train.corr().iloc[[1]]
X = np.asarray(df[['Pclass','Sex']])
y = np.asarray(df[['Survived']])
y.shape, X.shape
X = preprocessing.StandardScaler().fit(X).transform(X)
X
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver = 'liblinear', random_state = 42, max_iter = 1e3).fit(X,y)
LR
x_test = np.asarray(df_test[['Pclass','Sex']])
X_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
yhat = LR.predict(X_test)
yhat
result.head()
y_val = np.asarray(result[['Survived']]) 
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_val, yhat))
print(classification_report(y_val, yhat))
dict_ = {'PassengerId': df_test['PassengerId'], 'Survived': yhat }
final_result = pd.DataFrame(dict_)
final_result
final_result.to_csv('results.csv',index=False)