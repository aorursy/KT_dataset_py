#Imports

import pandas as pd

import numpy as np

from catboost import Pool, CatBoostClassifier, cv, CatBoostRegressor



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#Loading csv files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.describe()
#Separating label data for training

x = train.drop('target',axis=1)

y = train['target']
#Checking for any categorical features

cate_features_index = np.where(x.dtypes != float)[0]
cate_features_index
#make the x for train and test (also called validation data) 

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.99,random_state=1236)
#let us make the catboost model, use_best_model params will make the model prevent overfitting

model = CatBoostClassifier(iterations=1500, learning_rate=0.01, l2_leaf_reg=3.5, depth=8, rsm=0.98, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)
#now just to make the model to fit the data

model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
#last let us make the submission,note that you have to make the pred to be int!

pred = model.predict_proba(test)

preds= pred[:,1]
#generating submission csv

submission = pd.DataFrame({'ID_code':test['ID_code'],'target':preds})

#save the file to your directory

submission.to_csv('submission_prob.csv',index=False)