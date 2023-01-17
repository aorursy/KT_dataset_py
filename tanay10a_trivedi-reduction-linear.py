import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import random_projection
from sklearn.model_selection import train_test_split
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
trainX=pd.read_csv('../input/trainFeatures.csv')
trainy=pd.read_csv('../input/trainLabels.csv')
trainX_na=trainX.dropna(axis=1)
trainX_na=trainX_na.drop(['ids', 'RatingID', 'erkey', 'AccountabilityID', 'RatingYear'],axis=1)
trainX_na=trainX_na.drop(['BaseYear','RatingTableID','CNVersion','Tax_Year'],axis=1)
#smooth_vals=['Creation_Date', 'date_calculated', 'publish_date','Rpt_Comp_Date','EmployeeID','Rpt_Ap_Date','Rpt_Ap_Emp','Govt_Grants','Total_Contributions','ProgSvcRev','MemDues','PrimaryRev','Other_Revenue','Total_Revenue','Excess_Deficit','Total_Expenses','Program_Expenses','Administration_Expenses','Fundraising_Expenses','Pymt_Affiliates','Total_Assets','Total_Liabilities','Total_Net_Assets','Total_Func_Exp']

trainy=trainy['OverallScore']
trainy=trainy[np.isfinite(trainy)]
trainX_na_smooth=trainX_na.loc[trainy[np.isfinite(trainy)].index]
X_train, X_val, y_train, y_val = train_test_split(trainX_na_smooth, trainy, test_size=0.20, random_state=42)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
from sklearn import datasets, cluster
agglo = cluster.FeatureAgglomeration(n_clusters=10)
agglo.fit(X_train)
X_reduced = agglo.transform(X_train)
X_reduced.shape
lm1=LinearRegression()
lm1.fit(X_reduced,y_train)
preds=lm.predict(X_val)
rmse(preds,y_val)
X_reduced_val = agglo.transform(X_val)
preds=lm1.predict(X_reduced_val)
rmse(preds,y_val)
test=pd.read_csv("../input/testFeatures.csv")
test=test[X_train.columns]
X_trans = agglo.transform(test)
preds=lm1.predict(X_trans)
preds
out=pd.DataFrame({'Id':np.arange(1,2127),'OverallScore':preds})
out.to_csv('out_lm_transform.csv',index=False)