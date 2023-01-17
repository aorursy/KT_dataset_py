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
import pandas as pd
train = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv')
test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')
train = train.values[:,1:]
test = test.values[:,1:]

x = train[:,:-1]
y = train[:,[-1]]

test = test[:,:-1]
y.shape
from sklearn.preprocessing import LabelEncoder
import numpy as np

classle=LabelEncoder()
y=classle.fit_transform(y)
print(np.unique(y))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20,p=2)
knn.fit(x,y)
y_pred = knn.predict(test)
y_pred
submit = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')
for i in range(len(y_pred)):
  submit['Label'][i]=int(y_pred[i])

submit['Label']=submit['Label'].astype(int)
submit
submit.to_csv('submission_form.csv',index=False)