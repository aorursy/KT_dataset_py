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
from sklearn import datasets
from sklearn import linear_model

datadict = datasets.load_breast_cancer()
datadict.keys()
dict_keys=(['data','target','target_names','DESCR','feature_names','filename'])
X = datadict['data']
Y = datadict['target']

pd.DataFrame(X,columns=datadict['feature_names']).head()

## train test split 
from sklearn.model_selection import train_test_split

## 70% of Datafor training and 30% for Testing

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

print('Training Data:',X_train.shape,Y_train.shape)
print('Testing Data :',X_test.shape,Y_test.shape)
## ML model
model = linear_model.LogisticRegression()

#model.fit
model.fit(X_train,Y_train)

#model.predict
prediction=model.predict(X_test)

#accuracy
accuracy=np.mean(Y_test==prediction)
accuracy