# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
train = pd.read_csv('/kaggle/input/combined-train/combined_train.csv')
train.head()
test = pd.read_csv("/kaggle/input/combined-test/combined(od)_test.csv")
test.head()
test_id = test['Square_ID']
train = train.drop(['ee'], axis = 1)
test = test.drop(['ee'], axis = 1)
train.shape, test.shape
train['target_2015'].dtype
import matplotlib.pyplot as plt
plt.hist(train.target_2015, range=(0, 1), bins=50);
plt.title('target_2015');
train.columns
X=train.drop(['Square_ID','target_2015'], axis=1)
y = train['target_2015']
X.columns
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit_transform(X)
X.head()
#!pip install catboost
test =test.drop(['Square_ID'],axis =1)
from catboost import Pool, CatBoostClassifier, cv
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, log_loss
from sklearn.model_selection import KFold,StratifiedKFold, TimeSeriesSplit, GroupKFold
errcb=[]
y_pred_totcb=[]
fold=KFold(n_splits=12, shuffle=True,random_state=2001)
for train_index, test_index in fold.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    m = CatBoostClassifier(n_estimators = 2000,learning_rate=0.03,random_seed = 2001,eval_metric ='AUC') #, use_best_model=True)
    m.fit(X_train,y_train,eval_set=[(X_train,y_train)], early_stopping_rounds=300,verbose=100)
    preds=m.predict_proba(X_test)
    
    print("err: ",log_loss(y_test,preds))
    errcb.append(log_loss(y_test,preds))
    p = m.predict_proba(test)[:,1]
    y_pred_totcb.append(p)
pred =np.mean(y_pred_totcb,0)
np.mean(errcb)
d = {'ID': test_id, 'target': np.mean(y_pred_totcb, 0)}
sub = pd.DataFrame(data=d)
sub = sub[['ID', 'target']]
sub.to_csv("fresh.csv",index  = False)
sub.head()
# Feature Importance
fea_imp = pd.DataFrame({'imp':m.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))