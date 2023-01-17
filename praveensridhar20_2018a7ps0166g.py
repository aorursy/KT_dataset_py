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
        

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier 
import random 
random.seed(100)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/minor-project-2020/train.csv",header=0)
X=df.drop(['id','target'],axis=1)
y=df['target']
params={}
lr=LogisticRegression(max_iter=1500,class_weight="balanced",solver="newton-cg")
clf=GridSearchCV(lr,params,verbose=1,n_jobs=-1)
clf.fit(X,y)
df1=pd.read_csv("../input/minor-project-2020/test.csv")
X_test=df1.drop(['id'],axis=1)
y_pred_test=clf.predict_proba(X_test)
print (y_pred_test)
ids=df1['id']
ids=ids[:,np.newaxis].astype(int)
y_pred_test=y_pred_test[:,1]
y_pred_test=y_pred_test[:,np.newaxis]
res=np.concatenate([ids,y_pred_test],axis=1)
dff=pd.DataFrame(data=res,columns=["id","target"])
dff.to_csv("2018A7PS0166G.csv")
