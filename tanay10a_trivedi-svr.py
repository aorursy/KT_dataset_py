import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.pipeline import make_pipeline

from sklearn.svm import SVR
trainX=pd.read_csv('../input/trainFeatures.csv')
trainy=pd.read_csv('../input/trainLabels.csv')
from sklearn.model_selection import train_test_split
trainX_na=trainX.dropna(axis=1)
trainX_na=trainX_na.drop(['ids', 'RatingID', 'erkey', 'AccountabilityID', 'RatingYear'],axis=1)
trainX_na=trainX_na.drop(['BaseYear','RatingTableID','CNVersion','Tax_Year'],axis=1)
smooth_vals=['Creation_Date', 'date_calculated', 'publish_date','Rpt_Comp_Date','EmployeeID','Rpt_Ap_Date','Rpt_Ap_Emp','Govt_Grants','Total_Contributions','ProgSvcRev','MemDues','PrimaryRev','Other_Revenue','Total_Revenue','Excess_Deficit','Total_Expenses','Program_Expenses','Administration_Expenses','Fundraising_Expenses','Pymt_Affiliates','Total_Assets','Total_Liabilities','Total_Net_Assets','Total_Func_Exp']

trainy=trainy['OverallScore']
trainy=trainy[np.isfinite(trainy)]
trainX_na_smooth=trainX_na.loc[trainy[np.isfinite(trainy)].index]
X_train, X_val, y_train, y_val = train_test_split(trainX_na_smooth, trainy, test_size=0.20, random_state=42)
from sklearn.preprocessing import StandardScaler
mean_vals=X_train[smooth_vals].mean()
std_vals=trainX_na[smooth_vals].std()
X_train[smooth_vals]=(X_train[smooth_vals]-mean_vals)/std_vals
X_val[smooth_vals]=(X_val[smooth_vals]-mean_vals)/std_vals
svr=SVR(C= 10, epsilon= 0.5, gamma= 1e-07, kernel ='linear')
svr.fit(X_train,y_train)
preds=svr.predict(X_val)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse(preds,y_val)
test=pd.read_csv("../input/testFeatures.csv")
test=test[X_train.columns]
test[smooth_vals]=(test[smooth_vals]-mean_vals)/std_vals
preds=svr.predict(test)
out=pd.DataFrame({'Id':np.arange(1,2127),'OverallScore':preds})
out.to_csv('out_svr.csv',index=False)