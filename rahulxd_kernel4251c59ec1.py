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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.impute import KNNImputer
df_train=pd.read_csv('../input/isteml2020/train.csv')
df_test=pd.read_csv('../input/isteml2020/test.csv')
df_train=df_train.drop(columns=['x18'])
df_test=df_test.drop(columns=['x18'])
df_train.head()

df_train.describe()
df_train.isnull().sum()
df_train.shape
df_test.tail()
df_test.shape
df_test.isnull().sum()
X_train=df_train.loc[:,df_train.columns!='y']
X_train.shape

X_test=df_test.loc[:,:]
Y_train=df_train['y']
Y_train=Y_train.to_numpy()
combo=pd.concat(objs=[X_train,X_test])
combo.describe()
combo.nunique()
combo=pd.get_dummies(data=combo,columns=['x9','x16','x17','x19'],dummy_na=True,drop_first=True)

X_train_dummy=pd.DataFrame(data=combo_filled[0:Y_train.shape[0]])

X_train_dummy=scale(X_train_dummy)
X_test_dummy=pd.DataFrame(data=combo_filled[Y_train.shape[0]:])
X_test_dummy=scale(X_test_dummy)
X_train_filled=KNNImputer(n_neighbors=1000).fit_transform(X_train_dummy)
X_train_filled=pd.DataFrame(data=X_train_filled)
X_test_filled=KNNImputer(n_neighbors=500).fit_transform(X_test_dummy)
X_test_filled=pd.DataFrame(data=X_train_filled)
X_train_filled.columns=X_train_dummy.columns
X_test_filled.columns=X_test_dummy.columns
X_train=X_train_filled
X_test=X_test_filled

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train,Y_train) 
xgb=XGBClassifier(n_estimators=400,subsample=0.8,max_depth=3)
svm.fit(X_train,Y_train)
xgb.fit(X_train,Y_train)
predsvm=svm.predict(X_test)
predxgb=xgb.predict(X_test)
grid_predictions=grid.predict(X_test)
submit=pd.read_csv('../input/isteml2020/sampleSubmission.csv')

submit['Predicted']=pd.DataFrame(grid_predictions)
submit.to_csv('results_out.csv', index=False)

