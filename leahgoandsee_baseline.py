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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, auc
kickstart = pd.read_csv('../input/2020caohackathon/sample_set.csv')
kickstart.columns
kickstart.head()
# class ratio
print(kickstart['outcome'].value_counts())
print(kickstart['outcome'].value_counts()[0]/kickstart['outcome'].value_counts()[1])
# simple feature engineering
kickstart['create2launch']=(pd.to_datetime(kickstart['launched_at'])-pd.to_datetime(kickstart['created_at'])).dt.days
kickstart['launch2deadline']=(pd.to_datetime(kickstart['deadline'])-pd.to_datetime(kickstart['launched_at'])).dt.days

kickstart['staff_pick'] = kickstart['staff_pick'].astype(int)
kickstart['disable_communication'] = kickstart['disable_communication'].astype(int)

kickstart['create_hour'] = pd.to_datetime(kickstart['created_at']).dt.hour
kickstart['create_day'] = pd.to_datetime(kickstart['created_at']).dt.day
kickstart['create_month'] = pd.to_datetime(kickstart['created_at']).dt.month
kickstart['create_year'] = pd.to_datetime(kickstart['created_at']).dt.year
feature = ['goal', 'create2launch', 'launch2deadline',
           'staff_pick', 'disable_communication', 
           'create_hour', 'create_day', 'create_month', 'create_year',
           'launch_hour', 'launch_day', 'launch_month', 'launch_year',
           'deadline_hour', 'deadline_day', 'deadline_month', 'deadline_year']
kickstart[feature].head()
# impute missing value
data = kickstart[feature].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(data, kickstart['outcome'], test_size=0.2, stratify=kickstart['outcome'], random_state=42)  
print('Class 0/1 Ratio: {:.4} in train set.'.format(y_train.value_counts()[0]/y_train.value_counts()[1]))
print('Class 0/1 Ratio: {:.4} in test set.'.format(y_test.value_counts()[0]/y_test.value_counts()[1]))
# standardization
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)
lg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', intercept_scaling=1, max_iter=600, class_weight='balanced')
lg.fit(X_train, y_train) 
# cross validation
rskf = RepeatedStratifiedKFold(n_splits=6, n_repeats=3, random_state=42)
scores = cross_val_score(lg, X_train, y_train, scoring='roc_auc', cv=rskf, n_jobs=-1)
print('AUC: {:.4}'.format(scores.mean()))
# predict
y_pred = lg.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred[:,1]) 
roc_auc = auc(fpr,tpr)
plt.figure(figsize=(10,8))

plt.plot(fpr, tpr, color='deeppink', label=' ROC curve (area = %0.2f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
