import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("../input/heart-disease-uci/heart.csv")
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
print(df['sex'].value_counts())
sns.swarmplot(df['target'],df['age'],hue=df['sex'])
# For Heart Disease
df_1=df[(df['target']==1)]
print(df_1['sex'].value_counts())
sns.countplot(df_1['sex'])
# For No Heart Disease
df_0=df[(df['target']==0)]
print(df_0['sex'].value_counts())
sns.countplot(df_0['sex'])
plt.figure(figsize=(10,5))
sns.countplot(df['age'],hue=df['target'])
df_54=df[(df['age']<54)]
sns.countplot(df_54['target'])
df_55=df[(df['age']>55)]
sns.countplot(df_55['target'])
df['chol_cat']='NO'
# split the data using Q1,Q2,Q3,Q4
# Less than equal to 211-(1)
# greater than  equal to 212 and less than  equal to 240-(2)
# greater than  equal to 241 and less than  equal to 274-(3)
# greater than  equal to 275-(4)
df.loc[(df['chol']<=211),'chol_cat']='1'
df.loc[(df['chol']>=212) & (df['chol']<=240),'chol_cat']='2'
df.loc[(df['chol']>=241) & (df['chol']<=274),'chol_cat']='3'
df.loc[(df['chol']>=275),'chol_cat']='4'
sns.countplot(df['chol_cat'],hue=df['target'])
print(df['chol_cat'].value_counts())
# plt.xticks('0-211','212-240','241-274','275 and greater')
df['cp'].value_counts()
sns.countplot(df['cp'],hue=df['target'])
df['thalach'].describe()
df['heart_cat']='NO'
# split the data using Q1,Q2,Q3,Q4
# Less than 133.5-(1)
# greater than 133.5 and less than equal to 153-(2)
# greater than 153 and less than  equal to 166-(3)
# greater than 16675-(4)
df.loc[(df['thalach']<=133.5),'heart_cat']='1'
df.loc[(df['thalach']>133.5) & (df['thalach']<=153),'heart_cat']='2'
df.loc[(df['thalach']>153) & (df['thalach']<=166),'heart_cat']='3'
df.loc[(df['thalach']>166),'heart_cat']='4'
print(df['heart_cat'].value_counts())
sns.countplot(df['heart_cat'],hue=df['target'])
df['exang'].value_counts()
sns.countplot(df['exang'],hue=df['target'])
df.drop(['thalach','chol'],axis=1,inplace=True)
df['chol_cat']=df['chol_cat'].astype(int)
df['heart_cat']=df['heart_cat'].astype(int)
X=df.drop('target',axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
import statsmodels.api as sm
X_constant=sm.add_constant(X_train)
from statsmodels.stats.outliers_influence import variance_inflation_factor 
for i in zip(X_constant.columns,[variance_inflation_factor(X_constant.values,j) for j in range (0,X_constant.shape[1])]):
    print(i)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
lr=LogisticRegression(solver='liblinear')
lr.fit(X_train,y_train)
y_train_pred=lr.predict(X_train)
y_train_prob=lr.predict_proba(X_train)
y_train_prob=y_train_prob[:,1]
print("Accuracy score",accuracy_score(y_train,y_train_pred))
print('Confusion matrix-Train','\n',confusion_matrix(y_train,y_train_pred))
print('AUC - Train',roc_auc_score(y_train,y_train_prob))
y_test_predict=lr.predict(X_test)
y_test_prob=lr.predict_proba(X_test)[:,1]
print("Accuracy score",accuracy_score(y_test,y_test_predict))
print('Confusion matrix-Test','\n',confusion_matrix(y_test,y_test_predict))
print('AUC - Test',roc_auc_score(y_test,y_test_prob))
fpr,tpr,threashold=roc_curve(y_test,y_test_prob)

plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
# Here we are telling the model to create 100 decision tress and make random forest and calculated the score etc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
rfc=RandomForestClassifier(n_estimators=100,random_state=3)
params={'n_estimators':sp_randint(50,200),
        'max_features':sp_randint(1,24),
        'max_depth':sp_randint(2,10),
        'min_samples_split':sp_randint(2,20),
        'min_samples_leaf':sp_randint(1,20),
        'criterion':['gini','entropy']
}
rsearch_rfc=RandomizedSearchCV(rfc,param_distributions=params,cv=3,scoring='roc_auc',random_state=3,return_train_score=True)
rsearch_rfc.fit(X,y)
rsearch_rfc.best_estimator_
rsearch_rfc.best_params_
rfc=RandomForestClassifier(**rsearch_rfc.best_params_)
rfc.fit(X_train,y_train)
y_train_predict=rfc.predict(X_train)
y_train_prob=rfc.predict_proba(X_train)[:,1]
print("Accuracy score",accuracy_score(y_train,y_train_predict))
print("Confusion Matrix",confusion_matrix(y_train,y_train_predict))
print("Roc AUC score",roc_auc_score(y_train,y_train_prob))

y_test_predict=rfc.predict(X_test)
y_test_prob=rfc.predict_proba(X_test)[:,1]
print("Accuracy score",accuracy_score(y_test,y_test_predict))
print("Confusion Matrix",confusion_matrix(y_test,y_test_predict))
print("Roc AUC score",roc_auc_score(y_test,y_test_prob))
fpr,tpr,thresholds=roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
# important features
imp=pd.DataFrame(rfc.feature_importances_,index=X_train.columns,columns=['imp'])
imp=imp.sort_values(by='imp',ascending=False)
imp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from scipy.stats import randint as sp_randint
knc=KNeighborsClassifier()
params={
    'n_neighbors':sp_randint(2,6),
    'leaf_size':sp_randint(5,30),
    'p':sp_randint(1,5)
}
rs_knn=RandomizedSearchCV(knc,param_distributions=params,n_iter=15,scoring='roc_auc',return_train_score=True,cv=3)
rs_knn.fit(X,y)
rs_knn.best_params_
knc=KNeighborsClassifier(**rs_knn.best_params_)
knc.fit(X_train,y_train)

y_train_predict=knc.predict(X_train)
y_train_prob=knc.predict_proba(X_train)[:,1]
print("Accuracy score",accuracy_score(y_train,y_train_predict))
print("Confusion Matrix",confusion_matrix(y_train,y_train_predict))
print("Roc AUC score",roc_auc_score(y_train,y_train_prob))

y_test_predict=knc.predict(X_test)
y_test_prob=knc.predict_proba(X_test)[:,1]
print("Accuracy score",accuracy_score(y_test,y_test_predict))
print("Confusion Matrix",confusion_matrix(y_test,y_test_predict))
print("Roc AUC score",roc_auc_score(y_test,y_test_prob))
fpr,tpr,thresholds=roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier()
params={
    'n_estimators':sp_randint(5,40)
}
rs_ada=RandomizedSearchCV(ada,param_distributions=params,cv=3,scoring='accuracy')
rs_ada.fit(X,y)
rs_ada.best_params_
ada=AdaBoostClassifier(**rs_ada.best_params_)
ada.fit(X_train,y_train)
y_train_predict=ada.predict(X_train)
y_train_prob=ada.predict_proba(X_train)[:,-1]
y_train_predict
print('Overall Accuracy -Train',accuracy_score(y_train,y_train_predict))
print('Confusion matrix-Train','\n',confusion_matrix(y_train,y_train_predict))
print('AUC - Train',roc_auc_score(y_train,y_train_prob))
print("-------------------------------------------------------------------------")

y_test_pred=ada.predict(X_test)
y_test_prob=ada.predict_proba(X_test)[:,-1]
print('Overall Accuracy -Test',accuracy_score(y_test,y_test_pred))
print('Confusion matrix-Test','\n',confusion_matrix(y_test,y_test_pred))
print('AUC - Test',roc_auc_score(y_test,y_test_prob))
print("-------------------------------------------------------------------------")
print()

fpr,tpr,thresholds=roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
from sklearn.ensemble import VotingClassifier
lr=LogisticRegression(solver='liblinear')
knc=KNeighborsClassifier(**rs_knn.best_params_)
rfc=RandomForestClassifier(**rsearch_rfc.best_params_)
ada=AdaBoostClassifier(**rs_ada.best_params_)
clf=VotingClassifier(estimators=[('lr',lr),('knn',knc),('rfc',rfc),['ada',ada]],voting='hard')
clf.fit(X_train,y_train)
y_train_pred=clf.predict(X_train)
y_test_predt=clf.predict(X_test)

print('Accuracy score-Train',accuracy_score(y_train,y_train_pred))
print('Accuracy score-Test',accuracy_score(y_test,y_test_pred))


fpr,tpr,thresholds=roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
clf=VotingClassifier(estimators=[('lr',lr),('knn',knc),('rfc',rfc),['ada',ada]],voting='soft')
clf.fit(X_train,y_train)
y_train_pred=clf.predict(X_train)
y_test_predt=clf.predict(X_test)
y_train_prob=clf.predict_proba(X_train)[:,1]
y_test_prob=clf.predict_proba(X_test)[:,1]

print('Accuracy score-Train',accuracy_score(y_train,y_train_pred))
print('Accuracy score-Test',accuracy_score(y_test,y_test_pred))

print('Roc AuC Score-Train',roc_auc_score(y_train,y_train_prob))
print('Roc AuC Score-Test',roc_auc_score(y_test,y_test_prob))


fpr,tpr,thresholds=roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
