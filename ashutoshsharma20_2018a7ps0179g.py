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
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
print(df.shape)
df.head()
df = df.iloc[:, 1:]#removing id
print(df.shape)
df.head()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_test = scalar.transform(X_test)
params = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }
xgb = XGBClassifier(objective='binary:logistic')
#numFolds = 5
folds=2
#param_comb = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
#kfold_5 = cross_validate(xgb, scaled_X_train, y)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, error_score=0,
                                  cv=skf.split(scaled_X_train, y_train), verbose=3)
random_search.fit(scaled_X_train, y_train)
random_search.best_params_
y_pred = random_search.predict(scaled_X_test)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_pred))
df_test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")
scaled_test = scalar.transform(df_test.iloc[:, 1:])
print(scaled_test.shape)
res_test_proba = random_search.predict_proba(scaled_test)
mysub = pd.DataFrame({'id':df_test.id, 'target':res_test_proba[:,1]})
mysub.to_csv('sample_submission.csv', index=False)