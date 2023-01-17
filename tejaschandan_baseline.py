
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#IMPORT TRAIN DATA
df_train=pd.read_csv("/kaggle/input/isteml2020/train.csv")
df_train.head()
#IMPORT TEST DATA
df_test=pd.read_csv("/kaggle/input/isteml2020/test.csv")
df_test.head()
y_train=df_train['y']
y_train=y_train.to_numpy()
x_train=df_train.loc[:,df_train.columns!='y']
x_test=df_test

combo=pd.concat(objs=[x_train,x_test])
combo.describe()
combo.nunique()
combo=pd.get_dummies(data=combo ,columns=['x9','x16','x17','x18','x19'] , dummy_na=True,drop_first=True)

combo.describe()

x_train_dummy=pd.DataFrame(data=combo[0:y_train.shape[0]])
x_train_dummy.describe()
x_train_dummy_normalized=scale(x_train_dummy)
x_train_filled_final=KNNImputer(n_neighbors=5).fit_transform(x_train_dummy_normalized)
x_train_filled_final=pd.DataFrame(data=x_train_filled_final)
x_train_actual ,x_test_actual , y_train_actual , y_test_actual=train_test_split(x_train_dummy,y_train, test_size=.1)
x_test_dummy=pd.DataFrame(data=combo[y_train.shape[0]:])
x_test_dummy.describe()
x_test_predict=x_test_dummy
x_test_predict
x_train_normalized=scale(x_train_actual)
x_test_normalized=scale(x_test_actual)
x_test_normalized_predict=scale(x_test_predict)
x_train_try=pd.DataFrame(data=x_train_normalized)
x_train_try.head()
x_train_filled=KNNImputer(n_neighbors=5).fit_transform(x_train_normalized)
x_train_filled=pd.DataFrame(data=x_train_filled)
x_train_filled.isnull().sum()
x_train_filled.columns=x_train_dummy.columns
x_train_filled.head()
x_test_filled=KNNImputer(n_neighbors=5).fit_transform(x_test_normalized)
x_test_filled=pd.DataFrame(data=x_test_filled)
x_test_filled.columns=x_test_dummy.columns
x_test_filled.head()
x_train=x_train_filled
x_test=x_test_filled
x_test_filled_predict=KNNImputer(n_neighbors=5).fit_transform(x_test_normalized_predict)
x_test_filled_predict=pd.DataFrame(data=x_test_filled_predict)
log=LogisticRegression()
log.fit(x_train ,y_train_actual)
predlog=log.predict_proba(x_test)

print(roc_auc_score(y_test_actual,predlog[:,1]))
svmc=SVC(probability=True,random_state=2)
svmc.fit(x_train,y_train_actual)
predsvmc=svmc.predict_proba(x_test)

print(roc_auc_score(y_test_actual,predsvmc[:,1]))
rfc=RandomForestClassifier(n_estimators=600,random_state=2)
rfc.fit(x_train,y_train_actual)
predrfc=rfc.predict_proba(x_test)

print(roc_auc_score(y_test_actual,predrfc[:,1]))
xgb=XGBClassifier(n_estimators=500,subsample=0.9,colsample_bytree=0.9,max_depth=3,gamma=0,random_state=2)

params = {
    'subsample' : np.arange(0.7,0.8,0.01),
    'n_estimators': np.arange(500,800,100),
}
grid_xgb = GridSearchCV(estimator = xgb,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv=2,
                        verbose = 2,
                        n_jobs = -1)
grid_xgb.fit(x_train,y_train_actual)


print(grid_xgb.best_estimator_)
predxgb=xgb.predict_proba(x_test)
print(roc_auc_score(y_test_actual,predxgb[:,1]))
predf=predsvmc+predxgb+predrfc
predf=predf/3

print(roc_auc_score(y_test_actual,predf[:,1]))
#final=SVC(probability=True,random_state=2)
#final.fit(x_train_filled_final,y_train)
#finalpredsvmc=final.predict_proba(x_test)

#print(roc_auc_score(y_test_actual,finalpredsvmc[:,1]))
final=XGBClassifier(n_estimators=800,subsample=0.9,colsample_bytree=0.75,max_depth=3,gamma=0,random_state=2)
final.fit(x_train_filled_final,y_train)
pred=final.predict(x_test_filled_predict)
out=pd.DataFrame()
out["Id"]=range(0,4000)
out["Predicted"]=pd.DataFrame(pred[:,])
out.head()
out.to_csv('results_out.csv', index=False)
