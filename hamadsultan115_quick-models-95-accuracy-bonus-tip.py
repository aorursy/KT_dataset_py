# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/heart-failure-clinical-data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
clinical_records = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
clinical_records.head()
clinical_records.info()
corr_matrix = clinical_records.corr()
corr_matrix["DEATH_EVENT"].sort_values(ascending=False)
y = clinical_records["DEATH_EVENT"]
new_x =clinical_records[['age', 'ejection_fraction', 'serum_sodium', 'time']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size = 0.2, random_state = 2698)
X_train
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter = 1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, log_pred))
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=5)

forest_clf.fit(X_train, y_train)

forest_pred = forest_clf.predict(X_test)

accuracy_score(y_test, forest_pred)
from sklearn.ensemble import GradientBoostingClassifier
gradientboost_clf = GradientBoostingClassifier(max_depth = 1)
gradientboost_clf.fit(X_train,y_train)
gradientboost_pred = gradientboost_clf.predict(X_test)
accuracy_score(y_test,gradientboost_pred)
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
svc_pred = svm.predict(X_test)
accuracy_score(y_test,svc_pred)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(
 DecisionTreeClassifier(max_depth=1), n_estimators=200,
 algorithm="SAMME.R", learning_rate=0.3)
ada_clf.fit(X_train, y_train)

ada_pred = ada_clf.predict(X_test)

accuracy_score(y_test, ada_pred)