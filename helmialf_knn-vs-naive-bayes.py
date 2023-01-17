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
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from sklearn.naive_bayes import GaussianNB

import time
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

X = df.drop(['id','diagnosis','Unnamed: 32'],axis=1)

y = df.diagnosis

std_scaler = StandardScaler().fit(X)

X_std = std_scaler.transform(X)

X_std = pd.DataFrame(X_std, index=X.index, columns=X.columns)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_std,y,test_size=0.2,random_state=123)
clf_knn = KNeighborsClassifier()

scores = cross_val_score(clf_knn, X_train_1, y_train_1, cv=10)

print(scores)

print(scores.mean())

start_time = time.time()

clf_knn.fit(X_train_1, y_train_1)

preds = clf_knn.predict(X_train_1)

end_time = time.time()

print("Execution time : ",end_time - start_time)

print('Classification Report', classification_report(y_train_1,preds))

plot_confusion_matrix(clf_knn, X_train_1, y_train_1)
#split training data for naive bayes

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X,y,test_size=0.2,random_state=123)
clf_nb = GaussianNB()

scores = cross_val_score(clf_nb, X_train_2, y_train_2, cv=10)

print(scores)

print(scores.mean())
start_time = time.time()

clf_nb.fit(X_train_2, y_train_2)

preds = clf_nb.predict(X_train_2)

end_time = time.time()

print("Execution time : ",end_time - start_time)

print('Classification Report', classification_report(y_train_2,preds))

plot_confusion_matrix(clf_nb, X_train_2, y_train_2)