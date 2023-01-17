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
import seaborn as sns

train=pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv', index_col=0)

test=pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv', index_col=0)

submit=pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv', index_col=0)

train.head()
x=train.iloc[:,0:8]

y=train['8'].astype(np.int32) #예측 결과 정답을 저장할 y

print(x.shape)

print(y.shape)

print(test.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train_std=sc.fit_transform(x_train)

x_test_std=sc.transform(x_test)

test_std=sc.transform(test)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors = 2,p = 2)

knn.fit(x_train_std, y_train)
y_train_pred=knn.predict(x_train_std)

y_test_pred=knn.predict(x_test_std)

print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())

print('Misclassified test samples: %d' %(y_test!=y_test_pred).sum())
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_test_pred))
pred=knn.predict(test_std).astype(int)

for i in range(len(pred)):

    submit['Label'][i]=pred[i]

submit=submit.astype(np.int32)
submit.to_csv('submit.csv')