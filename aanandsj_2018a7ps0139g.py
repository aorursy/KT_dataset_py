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
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

%matplotlib inline

df = pd.read_csv('../input/minor-project-2020/train.csv')
df_test = pd.read_csv('../input/minor-project-2020/test.csv')


"""
from sklearn.model_selection import train_test_split

#X = new_train.drop(['target'],axis=1)
set=[0,2,3,7,10,11,13,14,15,16,17,18,22,24,26,27,33,35,52,55,56,57,58,60,62,63,64,65,66,67,69,70,71,72,73,75,76,77,78,79,81,82,83,84,85,86]
X = new_train.iloc[:,set]
y = new_train['target']

test_y=new_test.iloc[:,set]
#X=new_data.drop(["MEDV"])

X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state = 121)
"""
#X_train=df_train.drop(['id','target'],axis=1)
X_train=df.drop(['id','target'],axis=1)
Y_train=df['target']
X_train
X_test=df_test.drop(['id'],axis=1)
"""
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scaled_X_train=scalar.fit_transform(X_train)
scaled_X_val=scalar.fit_transform(X_val)
scaled_X_test=scalar.fit_transform(test_y)
"""
"""
from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=10,max_iter=100000).fit(scaled_X_train,y_train)
"""
"""
#dtclassifier=DecisionTreeClassifier()
#dtclassifier.fit(scaled_X_train,y_train)
"""
"""
y_pred=clf.predict(scaled_X_val)
#y_pred=clf.predict(scaled_X_test)
y_pred
"""
from sklearn.preprocessing import Normalizer
norm_X_train=Normalizer().fit_transform(X_train);
norm_X_test=Normalizer().transform(X_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(norm_X_train,Y_train)

y_pred=lr.predict_proba(norm_X_test)[:,1]

"""
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val,y_pred)
"""
df1=pd.DataFrame();
df1['id']=df_test['id']
df1['target']=y_pred
df1.to_csv('sub8.csv',index=False)#normalizeXtestbest
#df.to_csv('sub9.csv',index=False)#Xtest
#df.to_csv('sub10.csv',index=False)#selectcolumnsnormXtest


