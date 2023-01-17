# import libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline 
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/hr-analysis/train.csv')
test = pd.read_csv('../input/hr-analysis/test.csv')
sub = pd.read_csv('../input/hr-analysis/sample_submission.csv')


# replacing missing data by mode
for column in ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'experience','company_size',
              'company_type','last_new_job']:
    train[column].fillna(train[column].mode()[0], inplace=True)
    test[column].fillna(test[column].mode()[0], inplace=True)

	
#prepare features , target	
train_x = train.drop(columns=['target','enrollee_id'],axis=1)
train_y = train['target']

test = test.drop(columns='enrollee_id',axis=1)
#specifying categorical variables indexes
categorical_var = np.array([ 0, 2, 3,  4,  5,  6, 7,   8,  9, 10])

#fitting catboost classifier model
model = CatBoostClassifier(iterations=10000, learning_rate=0.001, l2_leaf_reg=3.5, depth=5, 
                           rsm=0.99, loss_function= 'Logloss')

model.fit(train_x,train_y,cat_features=categorical_var,verbose=True)

#predict probability on test data
predict_test = model.predict_proba(test)[:,1]

sub.target = predict_test
sub.to_csv('catboost.csv',index = False)