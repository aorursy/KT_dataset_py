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
from sklearn.metrics import roc_auc_score , roc_curve

from sklearn.model_selection import train_test_split 

from lightgbm import LGBMClassifier



import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/train.csv')

test = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/test.csv')
train.drop('id',axis=1,inplace=True)
def map_val(data):

    data["Gender"] = data["Gender"].replace({"Male":1, "Female":0})

    data["Vehicle_Age"] = data["Vehicle_Age"].replace({'> 2 Years':2, '1-2 Year':1, '< 1 Year':0 })

    data["Vehicle_Damage"] = data["Vehicle_Damage"].replace({"Yes":1, "No":0})

    return data



train = map_val(train)

test = map_val(test)
train.dtypes
test.dtypes
train.shape , test.shape
train['log_premium'] = np.log(train.Annual_Premium)

train['log_age'] = np.log(train.Age)

test['log_premium'] = np.log(test.Annual_Premium)

test['log_age'] = np.log(test.Age)
X = train.drop('Response',axis = 1)

y = train['Response']
X_t, X_tt, y_t, y_tt = train_test_split(X, y, test_size=.25, random_state=42)
train.columns
cat_col=['Gender','Driving_License', 'Region_Code', 'Previously_Insured', 

         'Vehicle_Age', 'Vehicle_Damage']
X_t.dtypes
lgbcl = LGBMClassifier(n_estimators=52)



lgbcl= lgbcl.fit(X_t, y_t,eval_metric='auc',eval_set=(X_tt , y_tt),verbose=2,categorical_feature=cat_col)



y_lgb = lgbcl.predict(X_tt)

probs_tr = lgbcl.predict_proba(X_t)[:, 1]

probs_te = lgbcl.predict_proba(X_tt)[:, 1]



print(roc_auc_score(y_t, probs_tr))

print(roc_auc_score(y_tt, probs_te))
feat_importances = pd.Series(lgbcl.feature_importances_, index=X_t.columns)

feat_importances.nlargest(15).plot(kind='barh')



plt.show()
test_df = test.drop('id',axis=1)
pred = lgbcl.predict_proba(test_df)[:,1]

submission = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/sample_submission.csv')
submission['Response'] = pred
submission.head()
submission.to_csv('light_gbm.csv',index=False)
## Got 85.67 in public leader board
from sklearn.model_selection import RandomizedSearchCV
param_grid={"boosting_type": ["gbdt","dart","goss","rf"],

           # "clf__class_weight":[None,"balanced"],

           #"clf__colsample_bytree": [.1,.2,.3,.4,.5,.6,.8,.9,1],

            "subsample":[.4,.5,.75,.9,1],

            "max_bin":[10,50,100,340,500,170],

            "importance_type":['split'],

            "num_leaves":[10,25,31,53,100,31,None],

            "n_estimators" : np.arange(10,200,10)

           #"clf__min_split_gain":[.05,.025,.01,.1],

            }
%%time

lgbm_model = RandomizedSearchCV(LGBMClassifier(),

                              param_distributions=param_grid,

                              n_iter=10,

                              cv = 5,

                              verbose=2,

                              scoring='roc_auc')



lgbm_model.fit(X_t, y_t,eval_metric='auc',eval_set=(X_tt , y_tt),verbose=2,categorical_feature=cat_col)
probs_tr = lgbm_model.predict_proba(X_t)[:, 1]

probs_te = lgbm_model.predict_proba(X_tt)[:, 1]



print(roc_auc_score(y_t, probs_tr))

print(roc_auc_score(y_tt, probs_te))
lgbm_model.best_score_
lgbm_model.best_params_
pred = lgbm_model.predict_proba(test_df)[:,1]
submission['Response'] = pred
submission.to_csv('lightgbm_random_cv.csv',index=False)