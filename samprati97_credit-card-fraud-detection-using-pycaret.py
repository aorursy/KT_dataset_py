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
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.describe()
data.info()
import seaborn as sns
sns.heatmap(data.corr())
sns.countplot('Class',data=data)
data['Class'].value_counts()
X = data.drop('Class',axis=1)
y = data['Class']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_test.shape, y_pred.shape
clf.score(X_test,y_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='1f')
accuracy_score(y_test,y_pred)
classification_report(y_test,y_pred)
!pip install pycaret
from pycaret.classification import *
clf1 = setup(data = data, target = 'Class')
# comparing all models
compare_models()
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
