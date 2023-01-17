import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import xgboost as xgb
%matplotlib inline
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
id_test = test_data['ID']
target = train_data['TARGET'].values
X_train = train_data.drop(['ID','TARGET'], axis=1)
X_test = test_data.drop(['ID'], axis=1).values
print ("The number of features before the domentionality reduction approach : ",X_train.shape[1])
clf = ExtraTreesClassifier()
clf = clf.fit(X_train,target)
clf.feature_importances_
model = SelectFromModel(clf,prefit=True)
Xr_Train = model.transform(X_train)
Xr_Test = model.transform(X_test)
print ("The number of features after the domentionality reduction approach : ",Xr_Test.shape[1])
clf = RandomForestClassifier(n_estimators=120, max_depth=17, random_state=1)
clf.fit(Xr_Train, target)
y_pred = clf.predict_proba(Xr_Test)
scores = cross_validation.cross_val_score(clf, Xr_Train, target, scoring='roc_auc', cv=5) 
print(scores.mean())
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission_rfc.csv", index=False)
xgbClassifier = xgb.XGBClassifier(n_estimators=580, max_depth=5, seed=1234, missing=np.nan, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, objective='binary:logistic') 
xgbClassifier.fit(Xr_Train,target)
y_xgb_pred = xgbClassifier.predict_proba(Xr_Test)
scores = cross_validation.cross_val_score(xgbClassifier, Xr_Train, target, scoring='roc_auc', cv=5) 
print(scores.mean())