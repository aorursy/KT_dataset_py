
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
from sklearn.linear_model import LogisticRegression
#IMPORT TRAIN DATA
df_train=pd.read_csv("/kaggle/input/isteml2020/train.csv")
df_train.head()
#IMPORT TEST DATA
df_test=pd.read_csv("/kaggle/input/isteml2020/test.csv")
df_test.head()
df_train.isnull().sum()
df_train.describe()
Y_train=df_train['y']
Y_train=Y_train.to_numpy()
print(Y_train.shape)
X_train= df_train.loc[:,df_train.columns!='y']
X_train.describe()
#print(X_train.shape)
X_test= df_test
X_test.describe()
#Combine Test and train data by concatenation
combo=pd.concat(objs=[X_train,X_test])
combo.describe()
#Detect categorical variables
combo.nunique()
#One-Hot-Encoding
combo=pd.get_dummies(data=combo,columns=["x9","x16","x17", "x18", "x19"],dummy_na=True,drop_first=True)
combo.head()
#Split Train and Test by index
X_train_dummy=pd.DataFrame(data=combo[0:Y_train.shape[0]])
X_train_dummy.describe()
X_test_dummy=pd.DataFrame(data=combo[Y_train.shape[0]:])
X_test_dummy.head()
#Normalise Data to improve performance of KNN
X_train_normalized = scale(X_train_dummy)
X_test_normalized=scale(X_test_dummy)
X_train_try=pd.DataFrame(data=X_train_normalized)
X_train_try.head()
#Imputing Misssing values with KNN
X_train_filled=KNNImputer(n_neighbors=5).fit_transform(X_train_normalized)
X_train_filled=pd.DataFrame(data=X_train_filled)

X_test_filled=KNNImputer(n_neighbors=5).fit_transform(X_test_normalized)
X_test_filled=pd.DataFrame(data=X_test_filled)
#Rename columns to orignal convention 
X_train_filled.columns=X_train_dummy.columns
X_train_filled.head()
X_test_filled.columns=X_test_dummy.columns
X_test_filled.head()
#Verify
X_train_filled.isnull().sum()
X_test_filled.isnull().sum()
#Final train and test after preprocessing is done
X_train=X_train_filled
X_test=X_test_filled
#Split train and 2nd test dataset
X_train, X_test2, Y_train,Y_test2 = train_test_split(X_train, Y_train, test_size=0.2, random_state=2)
#Logistic Regression

log=LogisticRegression(random_state=2)
log.fit(X_train,Y_train)
#SVM
svmc=SVC(probability=True,random_state=2)
svmc.fit(X_train,Y_train)
rfc1=RandomForestClassifier(criterion= 'gini', n_estimators=500,random_state=2)
rfc1.fit(X_train,Y_train)

rfc2=RandomForestClassifier(criterion= 'entropy', n_estimators=500,random_state=2)
rfc2.fit(X_train,Y_train)
#XGBoost
xgb=XGBClassifier(n_estimators=500,subsample=0.9,colsample_bytree=0.8,max_depth=3,gamma=0,random_state=2)
xgb.fit(X_train,Y_train)
predlog=log.predict_proba(X_test2)
predsvm=svmc.predict_proba(X_test2)
predrfc1=rfc1.predict_proba(X_test2)
predrfc2=rfc2.predict_proba(X_test2)
predxgb=xgb.predict_proba(X_test2)
pred2= (predlog+predsvm+predrfc1+predrfc2+predxgb)/5
print(pred2[0:100, :])
print(pred2.shape)
print(roc_auc_score(Y_test2,pred2[:,1]))


out=pd.DataFrame()
out["Id"]=range(0,4000)
out["Predicted"]=pd.DataFrame(pred[:, 0])
out.head(100)

out.to_csv('results_out3.csv', index=False)
