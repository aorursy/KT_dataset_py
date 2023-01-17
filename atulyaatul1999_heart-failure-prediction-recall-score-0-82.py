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
df=pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
df.shape
df.info()
df.corr()
fig = plt.figure(figsize=(22,20))

fig.add_axes([0,0,1,1])
ax = fig.get_axes()[0]
sns.heatmap(df.corr(), ax=ax, vmin=-1, vmax=1, annot=True)
df
y=df['DEATH_EVENT']
y
X=df.drop(columns=['DEATH_EVENT'])
X
X=X.values
# One row of data after Standardization
X[0]
from sklearn.utils import shuffle
X,y= shuffle(X,y, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
le=LogisticRegression()
le.fit(X_train,y_train)
print(le.intercept_,le.coef_)
y_pred=le.predict(X_test)
from sklearn.metrics import recall_score,precision_score,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
recall=recall_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
cm
# Confusion matrix 
sns.heatmap(cm, annot=True)
print(recall,precision)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=6)
dt.fit(X_train,y_train)
y_pred2=dt.predict(X_test)
recall_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
accuracy_score(y_test,y_pred2)
confusion_matrix(y_test,y_pred2)
sns.heatmap(confusion_matrix(y_test,y_pred2),annot=True)
