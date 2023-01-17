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
train = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv')

test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')

submission = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')

test.columns
train.head()
train_x = train.drop(['8','Unnamed: 0'],axis=1)

train_y = train['8']

test_x = test.drop(['8','Unnamed: 0'],axis=1)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(train_x,train_y, test_size=0.3, random_state=1,stratify=train_y)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=11, p=2)

knn.fit(X_train,y_train)

X_train.shape, X_test.shape, y_train.shape , y_test.shape
from sklearn.metrics import accuracy_score



X_test_pred = knn.predict(X_test)

X_train_pred = knn.predict(X_train)

print(accuracy_score(y_test,X_test_pred))

print(accuracy_score(y_train,X_train_pred))
X_train.shape , X_test.shape, y_train.shape, y_test.shape
test_y_pred = knn.predict(test_x)
submission.columns
submission['Label'] =test_y_pred

submission=submission.astype(np.int32)
submission.to_csv('submission_form.csv', mode='w', header= True, index= False)
submission