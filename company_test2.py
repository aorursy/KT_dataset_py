# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv")
data.head()
data.shot_made_flag.unique()
data.info()
data.lon.unique().shape
data_x=pd.get_dummies(data.action_type,prefix="action_type")
cols=["combined_shot_type","game_event_id","period","playoffs",
      "shot_type","shot_zone_area","shot_zone_basic","shot_zone_range",
      "matchup","opponent","game_date","shot_distance","minutes_remaining","seconds_remaining","loc_x","loc_y"]
for col in cols:
    data_x=pd.concat([data_x,pd.get_dummies(data[col],prefix=col),],axis=1)
train_x=data_x[-pd.isnull(data.shot_made_flag)]
test_x=data_x[pd.isnull(data.shot_made_flag)]
train_y=data.shot_made_flag[-pd.isnull(data.shot_made_flag)]
train_x.shape,train_y.shape
###尝试下用xgboost
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
clf = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=250,
                     subsample=0.5, colsample_bytree=0.5, seed=0)
clf.fit(train_x, train_y)
y_pred = clf.predict(train_x)
print("Number of mislabeled points out of a total %d points : %d"  % (train_x.shape[0],(train_y != y_pred).sum()))
###只包含字符型，准确率70%
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
clf = linear_model.LogisticRegression(C=1e5,penalty='l1',max_iter=500)
clf = RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_split=1, random_state=0)
clf.fit(train_x, train_y)

y_pred = clf.predict(train_x)
print("Number of mislabeled points out of a total %d points : %d"  % (train_x.shape[0],(train_y != y_pred).sum()))
import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
logloss(train_y,clf.predict_proba(train_x)[:,1])
test_y=clf.predict_proba(test_x)[:,1]
test_id=data[pd.isnull(data.shot_made_flag)]["shot_id"]
submission=pd.DataFrame({"shot_id":test_id,"shot_made_flag":test_y})
submission.to_csv("submissson.csv",index=False)
submission.head()
