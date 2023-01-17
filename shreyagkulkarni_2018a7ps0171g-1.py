# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy
import sklearn
%matplotlib inline
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
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
from sklearn.model_selection import train_test_split

#TODO
labels=df.loc[:,'target']
X_train=df.loc[:, df.columns != 'target']
y_train=labels
X_val=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_val = scalar.transform(X_val)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs',max_iter=4000)
balance = [ {0:1,1:530}]
param_grid = dict(class_weight=balance)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc',verbose=5)
grid_result = grid.fit(scaled_X_train, y_train)
y_pred=grid_result.predict_proba(scaled_X_val);
p=pd.DataFrame()
#p.columns=['id','target']
p['id']=X_val.iloc[:,0]
p['target']=y_pred[:,1]
print (len(p[(p['target']>0.5)]))
print (len(p[(p['target']<=0.5)]))
p.to_csv('grid.csv',index=False)