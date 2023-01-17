# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import math
diam = pd.read_csv("../input/diamonds.csv",index_col="Unnamed: 0")
onehot = ["cut","color","clarity"]
le = LabelEncoder()
ohe = OneHotEncoder()
lf = diam[onehot].astype(str).apply(lambda x:le.fit_transform(x))
sparse = ohe.fit_transform(lf)
ds = pd.DataFrame(sparse.toarray(),index = diam.index)
xdiam = pd.concat([diam,ds],axis=1)
xdiam.index = range(len(xdiam))
f = xdiam.columns
f = f[f.isin(onehot)==False].tolist()
f.remove("price")
# did I write nmse right?
def score(y_true,y_pred):
    e1 = mean_squared_error(y_true, y_pred)
    e0 = ((y_true**2).sum())/len(y_true)
    nmse = e1/e0
    return nmse

def run_lgb(train_X, train_y, val_X, val_y, params):
    print("lgb")
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=10, 
                      evals_result=evals_result)
    return model
train,test = train_test_split(xdiam,test_size=1000,random_state=42)
train,valid = train_test_split(train,test_size=1000,random_state=42)
train_X = train[f]
train_y = train["price"]
val_X = valid[f]
val_y = valid["price"] 
test_X = test[f]
test_y = test["price"]

params = {
        "n_jobs":4,
        "num_leaves":100,
        "learning_rate":0.03,
        "verbosity" : 1
    }
params["njob"] = 4
params["objective"] = "regression"
model = run_lgb(train_X, train_y, val_X, val_y, params)
pre = model.predict(test_X)
score(test_y,pre)
# just_wrote_a_simple_kfold
def lgb_kfold(X,y,params,k=10):
    kf = KFold(n_splits=k)
    folds = kf.split(X)
    vp = 0
    i = 0 
    xs = 0
    for train_index,val_index in folds:
        print(f"round:{i+1}")
        i += 1
        train_X,train_y = X[train_index],y[train_index]
        val_X,val_y = X[val_index],y[val_index]
        lmodel = run_lgb(train_X, train_y, val_X, val_y,params)
        lp = lmodel.predict(val_X)
        s = score(val_y,lp)
        xs += s
        print(s)
    
    vp = vp/k
    print("average:"+str(xs/k))
    return vp
X = np.array(xdiam[f])
y = xdiam["price"]
lgb_kfold(X,y,params,k=5)
