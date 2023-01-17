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
import numpy as np
import pandas as pd
#from fancyimpute import KNN
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
#from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('train.csv')
Y=df['y']
Y=Y.to_numpy()
X=df.loc[:, df.columns != 'y']
df1=pd.read_csv('test.csv')
combo = pd.concat(objs=[X,df1])
combo.nunique()
combo=pd.get_dummies(data=combo,columns=["x9","x16","x17","x18","x19"],dummy_na=True,drop_first=True)
combo.head()
X1=pd.DataFrame(data=combo[0:Y.shape[0]])
test=pd.DataFrame(data=combo[Y.shape[0]:])

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y,test_size=0.3,random_state=2)
X_train_normalized = scale(X_train)
X_test_normalized=scale(X_test)
X_train_try=pd.DataFrame(data=X_train_normalized)
X_train_try.head()
X_train_filled=KNNImputer(n_neighbors=5).fit_transform(X_train_normalized)
X_train_filled=pd.DataFrame(data=X_train_filled)
X_train_filled.columns=X_train.columns
X_train_filled.head()
X_test_filled=KNNImputer(n_neighbors=5).fit_transform(X_test_normalized)
X_test_filled=pd.DataFrame(data=X_test_filled)
X_test_filled.columns=X_test.columns
X_test_filled.head()

log=LogisticRegression(random_state=2)
log.fit(X_train_filled,Y_train)
predlog=log.predict_proba(X_test)

print(roc_auc_score(Y_test,predlog[:,1]))#accuracy_score(Y_test,predlog))
svmc=SVC(probability=True,random_state=2)
svmc.fit(X_train,Y_train)
predsvm=svmc.predict_proba(X_test)
print(roc_auc_score(Y_test,predsvm[:,1]))#accuracy_score(Y_test,predsvm))
rfc=RandomForestClassifier(n_estimators=500,random_state=2)
rfc.fit(X_train,Y_train)
predrfc=rfc.predict_proba(X_test)

print(roc_auc_score(Y_test,predrfc[:,1]))#,accuracy_score(Y_test,predrfc))
#xgb=XGBClassifier(n_estimators=500,subsample=0.9,colsample_bytree=0.8,max_depth=3,gamma=0,random_state=2)
#xgb.fit(X_train,Y_train)
#predxgb=xgb.predict_proba(X_test)
#print(roc_auc_score(Y_test,predxgb[:,1]))#,accuracy_score(Y_test,predxgb))
param = {
    'C': np.arange(1,52,0.5),
    'degree': np.arange(0,11,1),
    'gamma': ['scale','auto'],
    
    
}
#grid = GridSearchCV(SVC(),param,refit=True,verbose=2)
#grid.fit(X_train,Y_train)
#print(grid.best_estimator_)
X1_normalized = scale(X1)
X1_filled=KNNImputer(n_neighbors=5).fit_transform(X1_normalized)
X1_filled=pd.DataFrame(data=X1_filled)
test_normalised = scale(test)
test_filled=KNNImputer(n_neighbors=5).fit_transform(test_normalised)
test_filled=pd.DataFrame(data=test_filled)
svmc=SVC(probability=True,random_state=2,C=3.98,degree=0,gamma='auto')
svmc.fit(X1_filled,Y)
pred=svmc.predict(test_filled)
pred
out = pd.DataFrame()
out['Id'] = range(0,4000)
out['Predicted']=pd.DataFrame(pred[:,])
out.head()
out.to_csv('r.csv',index=False)
grid = GridSearchCV(SVC(),param,refit=True,verbose=2)
grid.fit(X1_filled,Y)
grid.best_params_

