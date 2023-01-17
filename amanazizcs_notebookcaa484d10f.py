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
import matplotlib.pyplot as plt
import seaborn as sns 

%matplotlib inline
df = pd.read_csv("../input/minor-project-2020/train.csv")
df.head()
X = df.drop('target', axis = 1)
X = X.drop('id', axis = 1)
#X = X.drop('col_39', axis = 1)
Y = df['target']
from sklearn.model_selection import train_test_split
#X_train_, X_test, y_train_, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
#X_train.shape
#X_test.shape
from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=27)
sm = SMOTE(random_state=27)
X_train, y_train = sm.fit_sample(X_train_, y_train_)
print(X_train.shape, y_train.shape)
x = 0
for i in y_train:
    x = x + i
print(x, y_train.shape[0]/2)
print(X_train.shape, X_test.shape)


X_train.describe()

X_train.head()
from sklearn.preprocessing import StandardScaler
X_train_scaled.shape
X_test = X_test.drop('col_39', axis = 1)

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)
#X_test.drop(['target'], axis = 1, inplace = True)
#print(X_test.shape, X_train.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)


res = lr.predict_proba(X_train_scaled)[:, 1]
#xgb = XGBClassifier()
#xgb.fit(X_train, y_train)
res = lr.predict_proba(X_test_scaled)[:,1]
res
from sklearn.metrics import roc_auc_score
auc_score1 = roc_auc_score(y_test, res)
print(auc_score1)
df2 = pd.read_csv("../input/minor-project-2020/test.csv")
df2.drop(['id'], axis = 1, inplace = True)
X_final = df2
#X_final = X_final.drop('col_39', axis = 1)
X_final.shape


X_final_scaled = scalar.transform(X_final)
X_final_scaled.shape
Y_final = lr.predict_proba(X_final_scaled)[:,1]
df2 = pd.read_csv("../input/minor-project-2020/test.csv")
a = df2['id']
ll = []
for i in range(a.shape[0]):
    ll.append([a[i],Y_final[i]])
pd.DataFrame(ll).to_csv("./answer.csv", header=['id','target'],index=None)

