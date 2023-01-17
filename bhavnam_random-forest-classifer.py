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
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
data.head()
data.describe()
import matplotlib.pyplot as plt
import seaborn as sns

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
data.diagnosis.value_counts()
sns.countplot(x = 'diagnosis', data = data, palette = 'hls')
plt.show()
y = data.diagnosis.values
x = data.drop(['diagnosis'], axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

#score=[]

for i in [10,50,100,200,500]:
    model = RandomForestClassifier(criterion="entropy", n_estimators=i)
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    print('No of estimators %d:'%i,accuracy_score(y_test,y_pred))
model = RandomForestClassifier(criterion="entropy", n_estimators=100)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
cm = confusion_matrix(y_test, y_pred)
cm
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = 2*(precision*recall)/(precision+recall)

print("F1 score:",f1_score)
from sklearn.metrics import roc_auc_score,roc_curve
y_pred = model.predict_proba(x_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()