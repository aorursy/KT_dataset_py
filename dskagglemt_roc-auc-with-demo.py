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

%matplotlib inline
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
X = df.iloc[:,0:-1].values

y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
from sklearn.svm import SVC

clf_svc = SVC(kernel = 'rbf', random_state = 1)

clf_svc.fit(X_train, y_train)



svc_y_pred = clf_svc.decision_function(X_test)

# svc_y_pred = clf_svc.predict(X_test)
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()



clf_lr.fit(X_train, y_train)



lr_y_pred = clf_lr.decision_function(X_test)

# lr_y_pred = clf_lr.predict(X_test)
from sklearn.metrics import roc_curve, auc



lr_fpr, lr_tpr, threshold = roc_curve(y_test, lr_y_pred)

lr_auc = auc(lr_fpr, lr_tpr)



svc_fpr, svc_tpr, threshold = roc_curve(y_test, svc_y_pred)

svc_auc = auc(svc_fpr, svc_tpr)



plt.figure(figsize = (5,5), dpi=100)

plt.plot(svc_fpr, svc_tpr, linestyle='-', label = "SVC (auc  = %0.3f)"%svc_auc)

plt.plot(lr_fpr, lr_tpr, marker='.', label = "Logistic (auc  = %0.3f)"%lr_auc)



plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')



plt.legend()



plt.show()