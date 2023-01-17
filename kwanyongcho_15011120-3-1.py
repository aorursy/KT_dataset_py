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



train_data=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv')

test_data=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')

submit=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')
#print(train_data.shape)

X=train_data.drop('8', axis=1)

#print(X.shape)

y=train_data['8']

#print(y.shape)
from sklearn.preprocessing import LabelEncoder

import numpy as np



classle=LabelEncoder()

y=classle.fit_transform(train_data['8'].values)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(X_train)

X_train_std=sc.transform(X_train)

X_test_std=sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

knn=KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train_std, y_train)

y_train_pred=knn.predict(X_train_std)

y_test_pred=knn.predict(X_test_std)

print("Fail training : %d" %(y_train!=y_train_pred).sum())

print("Fail testing : %d" %(y_test!=y_test_pred).sum())



print(accuracy_score(y_train, y_train_pred))

print(accuracy_score(y_test, y_test_pred))
X_submit_test=test_data.drop('8', axis=1)

X_submit_test_std=sc.transform(X_submit_test)

predict=knn.predict(X_submit_test_std)

for i in range(len(predict)):

    submit['Label'][i]=predict[i]



submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header=True, index=False)