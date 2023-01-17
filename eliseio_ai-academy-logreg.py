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
f = open("/kaggle/input/ai-academy-logreg/data_description.txt", "r")
print(f.read())
data = pd.read_csv('/kaggle/input/ai-academy-logreg/train.csv')
data.head()
data['y'] = data.y.astype('category').cat.codes
data.info()
data.poutcome.value_counts()
cat_features = ['job', 'marital', 'education', 'housing', 'loan','contact', 'month', 'previous', 'poutcome']

for i in cat_features:
    data[i] = data[i].astype('category').cat.codes
data.head()
data = data.drop('balance', axis=1)
data = data.drop('day', axis=1)
data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('y', axis=1), data['y'], test_size=0.3)
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred_train = LogReg.predict(X_train)
y_predict_proba_train = LogReg.predict_proba(X_train)[:,1]
LogReg.score(X_train, y_train)
from sklearn.metrics import roc_auc_score
print('ROC-AUC: ', roc_auc_score(y_train, y_predict_proba_train))
y_pred = LogReg.predict(X_test)
y_predict_proba = LogReg.predict_proba(X_test)[:,1]
LogReg.score(X_test, y_test)
print('ROC-AUC: ', roc_auc_score(y_test, y_predict_proba))