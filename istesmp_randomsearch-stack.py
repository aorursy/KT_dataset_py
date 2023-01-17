#IMPORTING ALL REQUIRED LIBRARIES

import numpy as np

import pandas as pd

#from fancyimpute import KNN

from sklearn.impute import KNNImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score,accuracy_score

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.preprocessing import scale

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression


df_train=pd.read_csv("/kaggle/input/isteml2020/train.csv")

df_test=pd.read_csv("/kaggle/input/isteml2020/test.csv")
df_train.head()
df_train.describe()
df_test.describe()
df_train.isnull().sum()
df_test.isnull().sum()
#Separate target varibale

Y_train=df_train['y']

Y_train=Y_train.to_numpy()

#Y_test=df_test['y']

#Y_test=Y_test.to_numpy()

#print(Y_train.shape,Y_test.shape)
#Separate features

X_train=df_train.loc[:, df_train.columns != 'y']

X_test=df_test.loc[:,df_test.columns!='y']
#Combine Test and train data by concatenation

combo=pd.concat(objs=[X_train,X_test])

combo.describe()
#Detect categorical variables

combo.nunique()
#One-Hot-Encoding

combo=pd.get_dummies(data=combo,columns=["x9","x16","x17","x18","x19"],dummy_na=True,drop_first=True)
combo.head()
combo.describe()
#Split Train and Test by index

X_train_dummy=pd.DataFrame(data=combo[0:Y_train.shape[0]])

X_train_dummy.describe()
X_test_dummy=pd.DataFrame(data=combo[Y_train.shape[0]:])

X_test_dummy.head()
#Normalise Data to improve performance of KNN

X_train_normalized = scale(X_train_dummy)

X_test_normalized=scale(X_test_dummy)

X_train_try=pd.DataFrame(data=X_train_normalized,columns=X_test_dummy.columns)

X_train_try.head()
#Imputing Misssing values with KNN

X_train_filled=KNNImputer(n_neighbors=7).fit_transform(X_train_normalized)

X_train_filled=pd.DataFrame(data=X_train_filled,columns=X_train_dummy.columns)
#Rename columns to orignal convention 

X_train_filled.columns=X_train_dummy.columns

X_train_filled.head()
#Verify

X_train_filled.isnull().sum()
X_test_filled=KNNImputer(n_neighbors=7).fit_transform(X_test_normalized)

X_test_filled=pd.DataFrame(data=X_test_filled,columns=X_test_dummy.columns)

X_test_filled.columns=X_test_dummy.columns

X_test_filled.head()
X_test_filled.isnull().sum()
#Final train and test after preprocessing is done

X_test=X_test_filled

#sm = SMOTE()

#X_train, Y_train = sm.fit_sample(X_train_filled, Y_train)
X_train=X_train_filled

#Logistic Regression



log=LogisticRegression(random_state=2)

log.fit(X_train,Y_train)
predlog=log.predict_proba(X_test)



#print(roc_auc_score(Y_test,predlog[:,1]))#accuracy_score(Y_test,predlog))
tuned_parameters_svc = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,'scale','auto'],

                     'C': [1, 10, 100, 1000]},

                   

md_svc=SVC()

svc_rscv=RandomizedSearchCV(md_svc,tuned_parameters_svc,n_iter=16,scoring='roc_auc',verbose=5,cv=3,n_jobs=-1)

svc_rscv.fit(X_train,Y_train)

svc_rscv.best_params_
#SVM



svmc=SVC(probability=True,random_state=2,gamma=0.001,C=1000)

svmc.fit(X_train,Y_train)
predsvm=svmc.predict_proba(X_test)

#print(roc_auc_score(Y_test,predsvm[:,1]))#accuracy_score(Y_test,predsvm))
tuned_parameters_rfc = {'n_estimators':[1000,2000,5000],'max_features':["log2","sqrt",1,0.5],

                     'max_depth': [2,3,4,5,6,10],'criterion':["gini"]}

md_rfc=RandomForestClassifier()

rfc_rscv=RandomizedSearchCV(md_rfc,tuned_parameters_rfc,n_iter=30,scoring='roc_auc',verbose=5,cv=3,n_jobs=-1)

rfc_rscv.fit(X_train,Y_train)

rfc_rscv.best_params_    
#Random Forest

rfc=RandomForestClassifier(n_estimators=2000,random_state=2,max_depth=10,max_features='sqrt')

rfc.fit(X_train,Y_train)
predrfc=rfc.predict_proba(X_test)



#print(roc_auc_score(Y_test,predrfc[:,1]))#,accuracy_score(Y_test,predrfc))


tuned_params_xgb={'n_estimators':[200],'subsample':[0.7],'colsample_bytree':[0.9],'max_depth':[7],'gamma':[1,5,10]}

md_xgb=XGBClassifier()

xgb_rscv=RandomizedSearchCV(md_xgb,tuned_params_xgb,n_iter=60,cv=3,scoring='roc_auc',verbose=5,n_jobs=-1)

xgb_rscv.fit(X_train,Y_train)

xgb_rscv.best_params_
#XGBoost

xgb=XGBClassifier(n_estimators=5000,subsample=0.7,colsample_bytree=0.9,max_depth=7,gamma=1,random_state=2)

xgb.fit(X_train,Y_train)
predxgb=xgb.predict_proba(X_test)

#print(roc_auc_score(Y_test,predxgb[:,1]))#,accuracy_score(Y_test,predxgb))
#Stacking weak models

predf=predxgb+predsvm

predf=predf/2







#print(roc_auc_score(Y_test,predf[:,1]))#,accuracy_score(Y_test,predf))




 



out=pd.DataFrame()

out["Id"]=range(0,4000)

out["Predicted"]=pd.DataFrame(predf[:,1])

out.head()



submission.head()
out.to_csv("Aimer.csv",index=False)