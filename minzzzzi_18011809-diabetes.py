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

import numpy as np
train = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv',index_col=0)

test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv',index_col = 0,usecols=[0,1,2,3,4,5,6,7,8])

submit = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')
train = np.array(train)

test = np.array(test)



x_train = train[:,0:8]

y_train = train[:,-1]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3, p = 1)

knn.fit(x_train,y_train)
test_pred = knn.predict(test)
for i in range(len(test)):

    submit['Label'][i] = test_pred[i]
submit['Label'] = submit['Label'].astype(int)
submit[:5]
submit.to_csv('submission.csv',index=False,header=True)