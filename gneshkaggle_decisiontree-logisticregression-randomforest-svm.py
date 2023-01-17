import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore')
df=pd.read_csv('Breast_cancer.csv')
df.head()
df.columns
df.info()
df.drop(['id','Unnamed: 32'],axis=1,inplace=True)
df.info()
df.shape
sns.distplot(df['radius_mean'])
sns.scatterplot(x=df['radius_mean'],y=df['radius_se'],hue=df['diagnosis'])
sns.scatterplot(x=df['texture_mean'],y=df['texture_se'],hue=df['diagnosis'])
sns.scatterplot(x=df['radius_mean'],y=df['radius_worst'],hue=df['diagnosis'])
plt.rcParams['figure.figsize']=(18,18)
df.hist();

corr=df.corr()
sns.heatmap(corr,fmt='.2f',annot=True,cmap=plt.cm.Blues)
from sklearn.model_selection import train_test_split
x=df.loc[:,df.columns!='diagnosis']
y=df.loc[:,'diagnosis']
x.shape,y.shape
#mapping the malignant as 1 and Benign as 0
y=y.map({'M':1,'B':0})
y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=12)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
dt=DecisionTreeClassifier(criterion='gini',max_depth=10)
dt.fit(x_train,y_train)
dt_pred=dt.predict(x_test)
accuracy_score(dt_pred,y_test)
print(confusion_matrix(dt_pred,y_test))
print(roc_auc_score(dt_pred,y_test))
params={'max_depth':np.arange(2,10),'min_samples_leaf':np.arange(2,10)}
dt_best=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params,cv=5,verbose=1)
dt_best.fit(x_train,y_train)
dt_best_pred=dt_best.predict(x_test)
accuracy_score(dt_best_pred,y_test)
dt_best.best_estimator_
dt_best.best_score_,dt_best.best_params_
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

lr_pred=lr.predict(x_test)
accuracy_score(lr_pred,y_test)
print(confusion_matrix(lr_pred,y_test))
lr_params={'C':[100,10,1,0.5,0.1,0.01],'penalty':['l1','l2']}
lr_best=GridSearchCV(estimator=LogisticRegression(),param_grid=lr_params,verbose=1,n_jobs=-1,cv=5)
lr_best.fit(x_train,y_train)
lr_best.best_params_,lr_best.best_score_
lr_best_pred=lr_best.predict(x_test)
accuracy_score(lr_best_pred,y_test)
print(confusion_matrix(lr_best_pred,y_test))
#Scaling the data 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
lr_best_scaled=lr_best.fit(x_train_scaled,y_train)
lr_best_scaled.best_params_,lr_best_scaled.best_score_
lr_best_scaled_pred=lr_best_scaled.predict(x_test_scaled)

means=lr_best_scaled.cv_results_['mean_test_score']
stds=lr_best_scaled.cv_results_['std_test_score']
params=lr_best_scaled.cv_results_['params']
for mean,std,param in zip(means,stds,params):
    print('%f %f in %r'%(mean,std,param))
accuracy_score(lr_best_scaled_pred,y_test)
print(confusion_matrix(lr_best_scaled_pred,y_test))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)
accuracy_score(rf_pred,y_test)
#training on the scaled data
rf.fit(x_train_scaled,y_train)
rf_pred_scaled=rf.predict(x_test_scaled)
accuracy_score(rf_pred_scaled,y_test)
rf_params={'max_features':[2,4,6,8,12,15,18,20,24,30],'n_estimators':[10,100,1000]}
rf_best=GridSearchCV(estimator=RandomForestClassifier(),param_grid=rf_params,cv=5,verbose=1,n_jobs=-1)
rf_best.fit(x_train,y_train)
rf_best_pred=rf_best.predict(x_test)
accuracy_score(rf_best_pred,y_test)
rf_best.best_params_,rf_best.best_score_
means=rf_best.cv_results_['mean_test_score']
stds=rf_best.cv_results_['std_test_score']
params=rf_best.cv_results_['params']
for mean,std,param in zip(means,stds,params):
    print('%f with std %f in %r'%(mean,std,param))
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
svc_pred=svc.predict(x_test)
accuracy_score(svc_pred,y_test)
svc.score(x_train,y_train)
#training on scaled data
svc.fit(x_train_scaled,y_train)
svc_pred_scaled=svc.predict(x_test_scaled)
accuracy_score(svc_pred_scaled,y_test)
svc.score(x_train_scaled,y_train)
svc.get_params().keys()
#hyperparameter tuning on scaled data
svc_params={'kernel':['linear','rbf','sigmoid','poly'],'C':[100,10,1,0.1,0.01,0.001]}
svc_best=GridSearchCV(estimator=SVC(),param_grid=svc_params,verbose=1,cv=5)
svc_best.fit(x_train_scaled,y_train)
svc_best_scaled_pred=svc_best.predict(x_test_scaled)
accuracy_score(svc_best_scaled_pred,y_test)
svc_best.best_params_,svc_best.best_score_






