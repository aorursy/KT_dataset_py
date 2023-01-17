!pip install xgboost



import numpy as np

import pandas as pd



from sklearn.datasets import load_iris

import xgboost as xgb

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

train.head()



train.target.value_counts()
train['sex'] = train['sex'].fillna('na')

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)



test['sex'] = test['sex'].fillna('na')

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)



train['sex'] = train['sex'].astype("category").cat.codes +1

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1



test['sex'] = test['sex'].astype("category").cat.codes +1

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1



train.head()
test.head()
x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]

y_train = train['target']



x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]



train_DMatrix = xgb.DMatrix(x_train, label= y_train)

test_DMatrix = xgb.DMatrix(x_test)
clf = xgb.XGBClassifier(n_estimators=3000, 

                        max_depth=18, 

                        learning_rate=0.15, 

                        num_class = 2, 

                        objective='multi:softprob',

                        seed=0,  

                        nthread=-1, 

                        scale_pos_weight = (32542./584.))



clf.fit(x_train, y_train)
clf.predict_proba(x_test)[:,1]



sub_xgb = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sub_xgb.target = clf.predict_proba(x_test)[:,1]
sub_new = pd.read_csv('../input/siimisicmysubmissions/sub-new.csv')

sub_mean = pd.read_csv('../input/siimisicmysubmissions/sub-mean.csv')



submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

submission.target = sub_mean.target *0.71 + sub_new.target *0.15 + sub_xgb.target *0.14

submission.head()
submission.to_csv('submission.csv', index = False)