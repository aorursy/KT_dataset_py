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
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

pd.set_option('display.max_columns', None)

df.head(10)
X = df.iloc[:, 1:-1].values

y = df.iloc[:, -1].values
import xgboost as xgb

clf = xgb.XGBClassifier(eta=0.2, n_estimators=400, max_depth=7, n_jobs=-1)
from imblearn.over_sampling import SMOTE, ADASYN

adasyn = ADASYN()

X_samp, y_samp = adasyn.fit_sample(X,y)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', solver='saga', max_iter=300, class_weight='balanced', n_jobs=-1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_samp = sc.fit_transform(X_samp)
lr.fit(X_samp, y_samp)
test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

testid = test['id'].values

X_test = test.iloc[:, 1:].values
X_test = sc.transform(X_test)
y_pred = lr.predict(X_test)
submission = pd.DataFrame({'id':testid, 'target':y_pred})
submission.to_csv('submission.csv', index=False)
clf.fit(X_samp, y_samp)
y_pred = lr.predict_proba(X_test)
y_pred
sub = y_pred[:, 1]
submission = pd.DataFrame({'id':testid, 'target':sub})

submission.to_csv('submission.csv', index=False)