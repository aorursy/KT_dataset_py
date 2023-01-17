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
import seaborn as sns
sns.heatmap(data.corr())
data.isnull().sum()
import matplotlib.pyplot as plt

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
data.diagnosis.value_counts()
sns.countplot(x = 'diagnosis', data = data, palette = 'hls')
plt.show()
y = data.diagnosis.values
x = data.drop(['diagnosis'], axis=1)
y.shape
from sklearn.model_selection import train_test_split
from sklearn import metrics

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier


k_range = range(1,26)
score= []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    score.append(metrics.accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt
plt.plot(k_range,score)
plt.xlabel('K neighbors')
plt.ylabel('Testing Accuracy')
#choosing 15 neighbors
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print("Accuracy:%f"%metrics.accuracy_score(y_test,y_pred))
confusion = metrics.confusion_matrix(y_test, y_pred)
confusion
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = 2*(precision*recall)/(precision+recall)

print("F1 score:",f1_score)
y_pred = knn.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()