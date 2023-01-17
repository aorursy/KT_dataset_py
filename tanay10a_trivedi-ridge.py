import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.linear_model import Ridge
trainX=pd.read_csv('../input/trainFeatures.csv')
trainy=pd.read_csv('../input/trainLabels.csv')
trainX_na=trainX.dropna(axis=1)
trainX_na=trainX_na.drop(['ids', 'RatingID', 'erkey', 'AccountabilityID', 'RatingYear'],axis=1)
trainX_na=trainX_na.drop(['BaseYear','RatingTableID','CNVersion','Tax_Year'],axis=1)
smooth_vals=['Creation_Date', 'date_calculated', 'publish_date','Rpt_Comp_Date','EmployeeID','Rpt_Ap_Date','Rpt_Ap_Emp','Govt_Grants','Total_Contributions','ProgSvcRev','MemDues','PrimaryRev','Other_Revenue','Total_Revenue','Excess_Deficit','Total_Expenses','Program_Expenses','Administration_Expenses','Fundraising_Expenses','Pymt_Affiliates','Total_Assets','Total_Liabilities','Total_Net_Assets','Total_Func_Exp']
trainy=trainy['OverallScore']
trainy=trainy[np.isfinite(trainy)]
trainX_na=trainX_na.loc[trainy[np.isfinite(trainy)].index]
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(trainX_na, trainy, test_size=0.20, random_state=42)
mean=X_train[smooth_vals].mean()
std=X_train[smooth_vals].std()
X_train[smooth_vals]=(X_train[smooth_vals]-mean)/std
X_val[smooth_vals]=(X_val[smooth_vals]-mean)/std
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
performance=pd.DataFrame(columns=['rmse'])
for i in np.linspace(start=0.01,stop=1,num=20):
    r=Ridge(alpha=i)
    r.fit(X_train,y_train)
    preds=r.predict(X_val)
    performance.loc[i,'rmse']=rmse(preds,y_val)
performance.sort_values('rmse',ascending=True)
r=Ridge(alpha=0.01)
r.fit(X_train,y_train)
test=pd.read_csv("../input/testFeatures.csv")
test=test[X_train.columns]
test[smooth_vals]=(test[smooth_vals]-mean)/std
preds=r.predict(test)
out=pd.DataFrame({'Id':np.arange(1,2127),'OverallScore':preds})
out.to_csv('out_ridge.csv',index=False)