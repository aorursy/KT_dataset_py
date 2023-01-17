import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.ensemble import RandomForestRegressor
trainX=pd.read_csv('../input/trainFeatures.csv')
trainy=pd.read_csv('../input/trainLabels.csv')
from sklearn.model_selection import train_test_split
trainX_na=trainX.dropna(axis=1)
trainX_na=trainX_na.drop(['ids', 'RatingID', 'erkey', 'AccountabilityID', 'RatingYear'],axis=1)
trainX_na=trainX_na.drop(['BaseYear','RatingTableID','CNVersion','Tax_Year'],axis=1)
trainy=trainy['OverallScore']
trainy=trainy[np.isfinite(trainy)]
trainX_na=trainX_na.loc[trainy[np.isfinite(trainy)].index]
X_train, X_val, y_train, y_val = train_test_split(trainX_na, trainy, test_size=0.20, random_state=42)
rf = RandomForestRegressor(bootstrap=True,max_depth=100,max_features='auto',min_samples_leaf=1,min_samples_split=2,n_estimators=1400,n_jobs=-1)
rf.fit(X_train,y_train)
preds=rf.predict(X_val)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse(preds,y_val)
test=pd.read_csv("../input/testFeatures.csv")
test=test[X_train.columns]
preds=rf.predict(test)
out=pd.DataFrame({'Id':np.arange(1,2127),'OverallScore':preds})
out.to_csv('out_rf.csv',index=False)