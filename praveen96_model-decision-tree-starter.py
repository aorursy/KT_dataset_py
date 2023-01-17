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
from sklearn.tree import DecisionTreeRegressor


#
original_train=pd.read_csv('../input/train.csv')
original_test=pd.read_csv('../input/test.csv')
print(original_train.head())
print(original_train.columns)
print(original_test.head())
decision_model_train_0=original_train.drop(['target'],axis=1)
target=original_train.target
print(target.head())

decision_model_train_1=decision_model_train_0.select_dtypes(exclude='object')
decision_model_test_1=original_test.select_dtypes(exclude='object')
model=DecisionTreeRegressor()
model.fit(decision_model_train_1,target)
predict_val=model.predict(decision_model_test_1)
model=DecisionTreeRegressor()
#decision_target_sub=pd.DataFrame({'ID' : original_test.ID, 'target' : predict_val})
#decision_target_sub.to_csv('decision_submit_target',index=False)
#decision_tree_score_model_2.08
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split



def get_mae(max_leaf_nodes,X,y):
    train_X,val_X,train_y,val_y=train_test_split(X,y)
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    predict_val=model.predict(val_X)
    mae=mean_absolute_error(predict_val,val_y)
    return(mae)
for leaf_nodes in [3,5,10,50]:
    mae=get_mae(leaf_nodes,decision_model_train_1,target)
    print("leaf nodes : %d mae : %d" % (leaf_nodes,mae))
#for max_leaf_nodes in [5,50,500,2,20,200,10,25]:
#    my_mae=get_mae(max_leaf_nodes,decision_model_train_1,target)
#    print("Max leaf nodes: %d \t\t Mean absolute error: %d" % (max_leaf_nodes,my_mae))
from sklearn.ensemble import RandomForestRegressor
def get_mae_forest(max_leaf_nodes,X,y):
    train_X,val_X,train_y,val_y=train_test_split(X,y)
    model=RandomForestRegressor()
    model.fit(train_X,train_y)
    predict_val=model.predict(val_X)
    mae=mean_absolute_error(predict_val,val_y)
    return(mae)
for leaf_nodes in [3,5,10,50]:
    mae=get_mae_forest(leaf_nodes,decision_model_train_1,target)
    print("leaf nodes : %d mae : %d" % (leaf_nodes,mae))

forest_model=RandomForestRegressor()
forest_model.fit(decision_model_train_1,target)
pred_val_forest=forest_model.predict(decision_model_test_1)




#forest_model_sub=pd.DataFrame({'ID' : original_test.ID,'target' : pred_val_forest})
#forest_model_sub.to_csv('forest_model_sub',index=False)

#forest_mpodel_score 1.73





from xgboost import XGBRegressor

#xgb_model=XGBRegressor()
#xgb_model.fit(decision_model_train_1,target,verbose=False)
#xgb_pred_val=xgb_model.predict(decision_model_test_1)
#xgb_model_sub=pd.DataFrame({'ID' : original_test.ID,'target' : xgb_pred_val})
#xgb_model_sub.to_csv('xgb_model_sub',index=False)
#xgb_my_model=XGBRegressor(n_estimators=1000,learning_rate=0.05)
#train_X,val_X,train_y,val_y=train_test_split(decision_model_train_1.as_matrix(),target.as_matrix())
#xgb_my_model.fit(train_X,train_y,early_stopping_rounds=5,
#                 eval_set=[(val_X,val_y)], verbose=False)
#xgb_pred_after=xgb_my_model.predict(decision_model_test_1.as_matrix())
#xgb_model_after_tun=pd.DataFrame({'ID' : original_test.ID,'target' : xgb_pred_after})
#xgb_model_after_tun.to_csv('xgb_model_after_tun',index=False)

#score _1.93
from xgboost import XGBRegressor

X=decision_model_train_1.as_matrix()
y=target
test=decision_model_test_1.as_matrix()


def get_mae(learning_rate,train_X,val_X,train_y,val_y):
    model=XGBRegressor(n_estimators=1000,learning_rate=learning_rate)
    model.fit(train_X,train_y,early_stopping_rounds=10,eval_set=[(val_X,val_y)],verbose=False)
    pred_val=model.predict(val_X)
    mae=mean_absolute_error(val_y,pred_val)
    return mae


train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.25)
for learning_rate in [0.1,0.2,0.5,1]:
    mae=get_mae(learning_rate,train_X,val_X,train_y,val_y)
    print("n_estimators = %d \t\t meanabsolute error = %d" % (learning_rate,mae))
    




