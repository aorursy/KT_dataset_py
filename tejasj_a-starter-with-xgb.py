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
datatrain = pd.read_csv("/kaggle/input/fetal-health-classification/fetal_health.csv")
datatrain.head()
datatrain.describe()
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
X=datatrain.drop("fetal_health",axis=1)
y = datatrain.fetal_health
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y.head()
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
ypred = xgb.predict(X_test)
#print(roc_auc_score(ypred,y_test))
print(f1_score(ypred,y_test,average='weighted'))
print(accuracy_score(ypred,y_test))