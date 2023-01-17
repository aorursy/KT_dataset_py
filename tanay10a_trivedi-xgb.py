import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from xgboost import XGBRegressor
trainX=pd.read_csv('../input/trainFeatures.csv')
trainy=pd.read_csv('../input/trainLabels.csv')
trainX_na=trainX.dropna(axis=1)
trainX_na=trainX_na.drop(['ids', 'RatingID', 'erkey', 'AccountabilityID', 'RatingYear'],axis=1)
trainX_na=trainX_na.drop(['BaseYear','RatingTableID','CNVersion','Tax_Year'],axis=1)
smooth_vals=['Creation_Date', 'date_calculated', 'publish_date','Rpt_Comp_Date','EmployeeID','Rpt_Ap_Date','Rpt_Ap_Emp','Govt_Grants','Total_Contributions','ProgSvcRev','MemDues','PrimaryRev','Other_Revenue','Total_Revenue','Excess_Deficit','Total_Expenses','Program_Expenses','Administration_Expenses','Fundraising_Expenses','Pymt_Affiliates','Total_Assets','Total_Liabilities','Total_Net_Assets','Total_Func_Exp']

trainy=trainy['OverallScore']
trainy=trainy[np.isfinite(trainy)]
trainX_na_smooth=trainX_na.loc[trainy[np.isfinite(trainy)].index]
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(trainX_na_smooth, trainy, test_size=0.20, random_state=42)
xgb=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=2.471198208376011,
       learning_rate=0.04593080232951641, max_delta_step=0, max_depth=48,
       min_child_weight=9, missing=None, n_estimators=234, n_jobs=1,
       nthread=-1, objective='reg:linear', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1) #fit with optimizer on my own server
xgb.fit(X_train,y_train)
preds=xgb.predict(X_val)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse(preds,y_val)
test=pd.read_csv("../input/testFeatures.csv")
test=test[X_train.columns]
preds=xgb.predict(test)
out=pd.DataFrame({'Id':np.arange(1,2127),'OverallScore':preds})
out.to_csv('out_xgb.csv',index=False)
