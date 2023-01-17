
 
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

#Logistic Regression

log=LogisticRegression(random_state=2)
log.fit(X_train,Y_train)

#SVM
svmc=SVC(probability=True,random_state=2)
svmc.fit(X_train,Y_train)

#Random Forest with ginin index
rfc1=RandomForestClassifier(criterion= 'gini', n_estimators=500,random_state=2)
rfc1.fit(X_train,Y_train)

#Random Forest with entropy
rfc2=RandomForestClassifier(criterion= 'entropy', n_estimators=500,random_state=2)
rfc2.fit(X_train,Y_train)
#XGBoost
xgb=XGBClassifier(n_estimators=500,subsample=0.9,colsample_bytree=0.8,max_depth=3,gamma=0,random_state=2)
xgb.fit(X_train,Y_train)

predlog_train=log.predict_proba(X_train)
predsvm_train=svmc.predict_proba(X_train)
predrfc1_train=rfc1.predict_proba(X_train)
predrfc2_train=rfc2.predict_proba(X_train)
predxgb_train=xgb.predict_proba(X_train)

print(predsvm_train.shape, predsvm_train[0:5, :], sep= '\n')

matrix_train= predlog_train[:, 0]
matrix_train= matrix_train.reshape(Y_train.shape[0],)
print(matrix_train.shape)
matrix_train= np.vstack((matrix_train, predsvm_train[:, 0].reshape(Y_train.shape[0],)))
matrix_train= np.vstack((matrix_train, predrfc1_train[:, 0].reshape(Y_train.shape[0],)))
matrix_train= np.vstack((matrix_train, predrfc2_train[:, 0].reshape(Y_train.shape[0],)))
matrix_train= np.vstack((matrix_train, predxgb_train[:, 0].reshape(Y_train.shape[0],)))

matrix_train= matrix_train.T
print(matrix_train.shape)

print(matrix_train[0:5, :])

matrix_train= pd.DataFrame(data= matrix_train)
matrix_train.head()

log_stacker=LogisticRegression(random_state=2)
log_stacker.fit(matrix_train,Y_train)

predlog_test=log.predict_proba(X_test)
predsvm_test=svmc.predict_proba(X_test)
predrfc1_test=rfc1.predict_proba(X_test)
predrfc2_test=rfc2.predict_proba(X_test)
predxgb_test=xgb.predict_proba(X_test)

matrix_test= predlog_test[:, 0]
matrix_test= matrix_test.reshape(4000,)
print(matrix_test.shape)
matrix_test= np.vstack((matrix_test, predsvm_test[:, 0].reshape(4000,)))
matrix_test= np.vstack((matrix_test, predrfc1_test[:, 0].reshape(4000,)))
matrix_test= np.vstack((matrix_test, predrfc2_test[:, 0].reshape(4000,)))
matrix_test= np.vstack((matrix_test, predxgb_test[:, 0].reshape(4000,)))

matrix_test= matrix_test.T
print(matrix_test.shape)

matrix_test= pd.DataFrame(data= matrix_test)
matrix_test.head()

pred= log_stacker.predict_proba(matrix_test)
print(pred[0:100, :])

print(pred.shape)

out=pd.DataFrame()
out["Id"]=range(0,4000)
out["Predicted"]=pd.DataFrame(pred[:, 1])
out.head(100)

pred_train=log_stacker.predict_proba(matrix_train)
print(roc_auc_score(Y_train,pred_train[:,1]))
 
out.to_csv('results_out6.csv', index=False)