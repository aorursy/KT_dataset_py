!pip install xgboost



import numpy as np

import pandas as pd



import xgboost as xgb

from sklearn.metrics import accuracy_score
train= pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test= pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sub   = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

# train.head()



# train.target.value_counts()
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')



test['sex'] = test['sex'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
train['sex'] = train['sex'].astype("category").cat.codes +1

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1

train.head()
test['sex'] = test['sex'].astype("category").cat.codes +1

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1

test.head()
x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]

y_train = train['target']





x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]

# y_train = test['target']





train_DMatrix = xgb.DMatrix(x_train, label= y_train)

test_DMatrix = xgb.DMatrix(x_test)
param = {

    'booster':'gbtree', 

    'eta': 0.3,

    'num_class': 2,

    'max_depth': 8

}



epochs = 100
clf = xgb.XGBClassifier(n_estimators=1000, 

                        max_depth=8, 

                        objective='multi:softprob',

                        seed=0,  

                        nthread=-1, 

                        learning_rate=0.015,

                        num_class = 2, 

                        scale_pos_weight = (32542/584))
clf.fit(x_train, y_train)

pred_train = clf.predict(x_train)
accuracy_score(pred_train, y_train)
# 0.9825212823763811 -- 0.015

# 0.9825212823763811-- 0.15
# proba = model.predict_proba(test_DMatrix) 

# clf.predict_proba(x_test)[:,1]

# clf.predict(x_test)

sub.target = clf.predict_proba(x_test)[:,1]

sub_tabular = sub.copy()
sub_public_merge = pd.read_csv('/kaggle/input/incredible-tpus-finetune-effnetb0-b6-at-once/submission_models_blended.csv')
sub.target = sub_public_merge.target *0.80 + sub_tabular.target *0.20
sub.to_csv('submission.csv', index = False)
sub.head(20)