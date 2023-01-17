# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the rxead-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



train = pd.read_csv("../input/logistic-classification-diabetes-knn/train.csv")

test = pd.read_csv("../input/logistic-classification-diabetes-knn/test_data.csv")
from sklearn.preprocessing import LabelEncoder 

import numpy as np

classle = LabelEncoder()

y = classle.fit_transform(train.iloc[:,-1].values)

print('species labels:', np.unique(y))
X_train = train.iloc[:,1:-1]

X_test = test.iloc[:,1:-1]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10, p = 2) 

knn.fit(X_train_std, y) 
y_train_pred=knn.predict(X_train_std)

y_test_pred=knn.predict(X_test_std)

print('Misclassified training samples: %d' %(y!=y_train_pred).sum())
submit = pd.read_csv("../input/logistic-classification-diabetes-knn/submission_form.csv")
for i in range(len(y_test_pred)):

  submit['Label'][i] = y_test_pred[i]

submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)