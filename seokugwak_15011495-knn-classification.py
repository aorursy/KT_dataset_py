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
test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')

test.drop( columns='Unnamed: 0', inplace=True)

test.head()
train = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv')

train.drop( columns='Unnamed: 0', inplace=True)

train.head()
form = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')

form.head()
dataset = train.values

X = dataset[:,0:8]

y = dataset[:,8]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=30, p=1)

knn.fit(X_train,y_train)



y_train_pred = knn.predict(X_train)

y_test_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_test_pred))
X_pred = test.values[:,0:8]

y_pred = knn.predict(X_pred)

form['Label']=y_pred.astype(int)

form.head()
form.to_csv('/kaggle/working/submission.csv',index=False)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier

knn1=KNeighborsClassifier(n_neighbors=23, p=1)

knn1.fit(X_train_std,y_train)
y_train_std_pred = knn1.predict(X_train_std)

y_test_std_pred = knn1.predict(X_test_std)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_test_std_pred))
X_std_pred = sc.transform(test.values[:,0:8])

y_std_pred = knn.predict(X_std_pred)

form['Label']=y_pred.astype(int)

form
form.to_csv('/kaggle/working/std_submission.csv',index=False)