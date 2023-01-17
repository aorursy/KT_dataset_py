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
data=pd.read_csv('/kaggle/input/data.csv')
data.head()
data_features=['day','month','deaths']
X=data[data_features]
X.head()
y=data.cases ## data to be predicted
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
X_train=preprocessing.normalize(X_train,norm='l1')
X_test=preprocessing.normalize(X_test,norm='l1')
forest_model = DecisionTreeRegressor(max_leaf_nodes=26,random_state=1)
forest_model.fit(X_train,y_train)
melb_preds = forest_model.predict(X_test)
print(melb_preds) ## predicted cases

print(y_test) ## Actual number of cases