import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_id = test.ID
print(train.shape , "Train shape")
print(test.shape , "Test shape")
train.head()
Y = train.target
Y = np.log(Y+1)

train = train.drop(['ID','target'], axis = 1)
test = test.drop(['ID'], axis = 1)
# taken from https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-santander-value
unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape
train = train.drop(constant_df["col_name"], axis=1)
test = test.drop(constant_df["col_name"], axis=1)
total = (train == 0).sum().sort_values(ascending = False)
percent = ((train == 0).sum()/(train==0).count()*100).sort_values(ascending = False)
train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
train_data.tail(60)
use_cols = train_data[train_data.Percent<80]
use_cols.shape
use_cols = use_cols.index
use_cols.shape
use_cols
train = train[use_cols]
test = test[use_cols]
print(train.shape , "Train shape")
print(test.shape , "Test shape")
%%time
def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


kf = model_selection.KFold(n_splits=10, shuffle=True)
def runRF(x_train, y_train,x_test, y_test,test):
    model=RandomForestRegressor(bootstrap=True, max_features=0.75, min_samples_leaf=11, min_samples_split=13, n_estimators=100)
    model.fit(x_train, y_train)
    y_pred_train=model.predict(x_test)
    mse=rmsle(np.exp(y_pred_train)-1,np.exp(y_test)-1)
    y_pred_test=model.predict(test)
    return y_pred_train,mse,y_pred_test

pred_full_test_RF = 0    
rmsle_RF_list=[]

for dev_index, val_index in kf.split(train):
    dev_X, val_X = train.loc[dev_index], train.loc[val_index]
    dev_y, val_y = Y.loc[dev_index], Y.loc[val_index]
    ypred_valid_RF,rmsle_RF,ytest_RF=runRF(dev_X, dev_y, val_X, val_y,test)
    print("fold_ RF _ok "+str(rmsle_RF))
    rmsle_RF_list.append(rmsle_RF)
    pred_full_test_RF = pred_full_test_RF + ytest_RF
    
rmsle_RF_mean=np.mean(rmsle_RF_list)
print("Mean cv score : ", np.mean(rmsle_RF_mean))
ytest_RF=pred_full_test_RF/10


ytest_RF = np.exp(ytest_RF)-1
out_df = pd.DataFrame(ytest_RF)
out_df.columns = ['target']
out_df.insert(0, 'ID', test_id)
out_df.to_csv("RF_" + str(rmsle_RF_mean) + "_.csv", index=False)

