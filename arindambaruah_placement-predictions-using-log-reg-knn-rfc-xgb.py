import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df_copy=df.copy()
df['sl_no'].unique().size
df.info()
df['salary'].fillna(0,inplace=True)

df['salary'].isna().any()
df_copy['salary'].median()
df['gender']=df['gender'].map({'M':0,'F':1})
df.head()
df['ssc_b'].value_counts()
df['ssc_b']=df['ssc_b'].map({'Central':1,'Others':0})
df['hsc_b']=df['hsc_b'].map({'Central':1,'Others':0})
df['hsc_s'].value_counts()
df_subjects=pd.get_dummies(df['hsc_s'])

df=df.merge(df_subjects,on=df.index)
df.drop('key_0',axis=1,inplace=True)

df.drop('hsc_s',axis=1,inplace=True)
df.columns
df=df[['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'Arts', 'Commerce','Science','degree_p',

       'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p', 'status',

       'salary']]
df['degree_t'].value_counts()
df_deg=pd.get_dummies(df['degree_t'])

df=df.merge(df_deg,on=df.index)
df.drop('key_0',axis=1,inplace=True)

df.drop('degree_t',axis=1,inplace=True)
df.columns
df=df[['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'Arts',

       'Commerce', 'Science', 'degree_p','Comm&Mgmt', 'Others',

       'Sci&Tech','workex', 'etest_p',

       'specialisation', 'mba_p', 'status', 'salary']]
df
df['workex']=df['workex'].map({'Yes':1,'No':0})

df.head()
df['specialisation'].value_counts()
df['specialisation']=df['specialisation'].map({'Mkt&Fin':1,'Mkt&HR':0})
df['status'].value_counts()
df['status']=df['status'].map({'Placed':1,'Not Placed':0})
df.head()


df.loc[df['ssc_p']<=60,'ssc_p_c']=3

df.loc[(df['ssc_p']>60) & (df['ssc_p']<81),'ssc_p_c']=2

df.loc[(df['ssc_p']>80)& (df['ssc_p']<101),'ssc_p_c']=1



df.loc[df['hsc_p']<=60,'hsc_p_c']=3

df.loc[(df['hsc_p']>60) & (df['hsc_p']<81),'hsc_p_c']=2

df.loc[(df['hsc_p']>80)& (df['hsc_p']<101),'hsc_p_c']=1





df.loc[df['degree_p']<=60,'degree_p_c']=3

df.loc[(df['degree_p']>60) & (df['degree_p']<81),'degree_p_c']=2

df.loc[(df['degree_p']>80)& (df['degree_p']<101),'degree_p_c']=1



df.loc[df['mba_p']<=60,'mba_p_c']=3

df.loc[(df['mba_p']>60) & (df['mba_p']<81),'mba_p_c']=2

df.loc[(df['mba_p']>80)& (df['mba_p']<101),'mba_p_c']=1



df.loc[df['etest_p']<=60,'etest_p_c']=3

df.loc[(df['etest_p']>60) & (df['etest_p']<81),'etest_p_c']=2

df.loc[(df['etest_p']>80)& (df['etest_p']<101),'etest_p_c']=1







qual_type=['ssc_p','hsc_p','degree_p','mba_p','etest_p']



for qual in qual_type:

    df.drop(qual,axis=1,inplace=True)





df.columns
df=df[['sl_no', 'gender', 'ssc_b', 'hsc_b','ssc_p_c', 'hsc_p_c', 'degree_p_c', 'mba_p_c', 'etest_p_c', 'Arts', 'Commerce', 'Science',

       'Comm&Mgmt', 'Others', 'Sci&Tech', 'workex', 'specialisation', 'status',

       'salary']]
df.head()
sns.catplot('gender',data=df_copy,kind='count',hue='status',palette='rocket')
sns.factorplot('gender',data=df_copy,kind='count',palette='winter')

plt.title('M/F ratio={0:.2f}'.format(df_copy['gender'].value_counts()[0]/df_copy['gender'].value_counts()[1]))
df_male=df_copy[df_copy['gender']=='M']
df_male['status'].value_counts()
male_placed_ratio=df_male['status'].value_counts()[0]/df_male['status'].value_counts()[1]
print('Placement ratio of male candidates:{0:.2f}'.format(male_placed_ratio))
df_female=df_copy[df_copy['gender']=='F']

df_female['status'].value_counts()
female_placed_ratio=df_female['status'].value_counts()[0]/df_female['status'].value_counts()[1]

print('Placement ratio of female candidates:{0:.2f}'.format(female_placed_ratio))
sns.catplot('ssc_p_c',data=df,kind='count')

plt.title('10th percentage distribution')

plt.xlabel('10th percentage class')
sns.catplot('ssc_p_c',data=df,kind='count',hue='status',palette='inferno')

plt.title('10th Percentage with placed/unplaced')

plt.xlabel('10th Percentage class')
sns.catplot('hsc_p_c',data=df,kind='count')

plt.title('12th percentage distribution')

plt.xlabel('12th Percentage class')
sns.catplot('hsc_p_c',data=df,kind='count',hue='status',palette='summer')

plt.title('12th Percentage with placement status')

plt.xlabel('12th Percentage class')
fig=plt.figure(figsize=(10,5))



ax1 = fig.add_subplot(121)



g = sns.countplot("degree_p_c" , data=df, ax=ax1,palette='ocean')



ax2=fig.add_subplot(122)



g=sns.countplot('degree_p_c',data=df,ax=ax2,hue='status',palette='winter')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('UG percentage distribution')

ax1.set_xlabel('UG percentage class')

ax2.set_title('UG percentage with placement status')

ax2.set_xlabel('UG percentage class')
fig1=plt.figure(figsize=(10,5))



ax1 = fig1.add_subplot(121)



g = sns.countplot("mba_p_c" , data=df, ax=ax1,palette='rocket')



ax2=fig1.add_subplot(122)



g=sns.countplot('mba_p_c',data=df,ax=ax2,hue='status',palette='viridis')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('MBA percentage distribution')

ax1.set_xlabel('MBA percentage class')

ax2.set_title('MBA percentage with placement status')

ax2.set_xlabel('MBA percentage class')
fig2=plt.figure(figsize=(10,5))



ax1 = fig2.add_subplot(121)



g = sns.countplot("etest_p_c" , data=df, ax=ax1,palette='PuRd')



ax2=fig2.add_subplot(122)



g=sns.countplot('etest_p_c',data=df,ax=ax2,hue='status',palette='BuPu')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('E test percentage distribution')

ax1.set_xlabel('E test percentage class')

ax2.set_title('E test percentage with placement status')

ax2.set_xlabel('E test precentage class')
fig3=plt.figure(figsize=(10,5))



ax1 = fig3.add_subplot(121)



g = sns.countplot("hsc_s", data=df_copy, ax=ax1,palette='OrRd')



ax2=fig3.add_subplot(122)



g=sns.countplot('hsc_s',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('High school specialisation distribution')

ax1.set_xlabel('HS specialisation')

ax2.set_title('High school spcialisation distribution')

ax2.set_xlabel('HS specialisation')
fig4=plt.figure(figsize=(10,5))



ax1 = fig4.add_subplot(121)



g = sns.countplot("specialisation" , data=df_copy, ax=ax1,palette='OrRd')



ax2=fig4.add_subplot(122)



g=sns.countplot('specialisation',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('MBA specialisation distribution')

ax1.set_xlabel('MBA specialisation')

ax2.set_title('MBA spcialisation distribution')

ax2.set_xlabel('MBA specialisation')
fig5=plt.figure(figsize=(10,5))



ax1 = fig5.add_subplot(121)



g = sns.countplot("degree_t" , data=df_copy, ax=ax1,palette='OrRd')



ax2=fig5.add_subplot(122)



g=sns.countplot('degree_t',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('UG specialisation distribution')

ax1.set_xlabel('UG specialisation')

ax2.set_title('UG spcialisation distribution')

ax2.set_xlabel('UG specialisation')
fig6=plt.figure(figsize=(10,5))



ax1 = fig6.add_subplot(121)



g = sns.countplot("ssc_b" , data=df_copy, ax=ax1,palette='rocket')



ax2=fig6.add_subplot(122)



g=sns.countplot('ssc_b',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('10th board students distribution')

ax1.set_xlabel('Students')

ax2.set_title('10th board students distribution')

ax2.set_xlabel('Students')
fig7=plt.figure(figsize=(10,5))



ax1 = fig7.add_subplot(121)



g = sns.countplot("hsc_b" , data=df_copy, ax=ax1,palette='rocket')



ax2=fig7.add_subplot(122)



g=sns.countplot('hsc_b',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('12th board students distribution')

ax1.set_xlabel('Students')

ax2.set_title('12th board students distribution')

ax2.set_xlabel('Students')
plt.figure(figsize=(10,8))

df_placed=df[df['salary']>0]

sns.kdeplot(df_placed['salary'],shade=True)

plt.xlabel('Salary (in Rs)',size=10)

plt.title('Salary distribution for the batch',size=15)
mean=df_copy['salary'].mean()

median=df_copy['salary'].median()
plt.figure(figsize=(10,8))

df_placed=df[df['salary']>0]

sns.distplot(df_placed['salary'])

plt.xlabel('Salary (in Rs)',size=10)

plt.title('Salary distribution for the batch',size=15)

plt.axvline(mean,color='red')

plt.axvline(median,color='green')

plt.title('Mean={0:.2f}   Median={1:.2f}'.format(mean,median))

corr=df.corr()
plt.figure(figsize=(20,10))

sns.heatmap(corr,annot=True,cmap='viridis')
unn_feat=['sl_no','ssc_b','hsc_b','salary']



for feat in unn_feat:

    df.drop(feat,axis=1,inplace=True)
df.head()
df.isna().any()
from sklearn.model_selection import train_test_split



Y=df['status'] #Labels

X=df.drop('status',axis=1) #Input data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0,shuffle=True)
from sklearn.neighbors import KNeighborsClassifier
k=[5,6,7,8,9,10,11,12,13,14,15]

scores=[]

for val in k:

    knn=KNeighborsClassifier(n_neighbors=val)

    knn.fit(X_train,y_train)

    scores.append(knn.score(X_train,y_train))

scores    
plt.figure(figsize=(10,8))

plt.plot(k,scores,color='green')

plt.xlabel('K neighbors',size=15)

plt.ylabel('Accuracy',size=15)

plt.xticks(np.arange(5,16))

plt.axvline(8,color='red')

knn=KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)
knn.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

labels=['Unplaced','Placed']

conf_mat_knn=confusion_matrix(y_test,y_pred_knn)

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = sns.heatmap(conf_mat_knn,ax=ax,annot=True,cmap='inferno')

ax.xaxis.set_ticklabels(['Unplaced', 'Placed'])

ax.yaxis.set_ticklabels(['Unplaced', 'Placed'],rotation=0)

ax.set_xlabel('Predicted',size=15)

ax.set_ylabel('Actual',size=15)

plt.title('Confusion matrix KNN',size=15)

plt.figure(figsize=(10,8))

from sklearn.linear_model import LogisticRegression
reg_log=LogisticRegression()
reg_log.fit(X_train,y_train)
reg_log.score(X_train,y_train)
y_pred_log=reg_log.predict(X_test)

reg_log.score(X_test,y_test)
conf_mat_log=confusion_matrix(y_pred_log,y_test)

labels=['Unplaced','Placed']

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = sns.heatmap(conf_mat_log,ax=ax,annot=True,cmap='viridis')

ax.xaxis.set_ticklabels(['Unplaced', 'Placed'])

ax.yaxis.set_ticklabels(['Unplaced', 'Placed'],rotation=0)

ax.set_xlabel('Predicted',size=15)

ax.set_ylabel('Actual',size=15)

plt.title('Confusion matrix Logistic Regression',size=15)

plt.figure(figsize=(10,8))
y_lr=reg_log.fit(X_train,y_train).decision_function(X_test)
from sklearn.metrics import roc_curve,auc,precision_recall_curve



fpr,tpr,_=roc_curve(y_test,y_lr)

plt.plot(fpr,tpr,color='indianred')

plt.plot([0,1],[0,1],linestyle='--')

auc_reg=auc(fpr,tpr).round(2)

plt.title('ROC curve with AUC={}'.format(auc_reg))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')
precision,recall,threshold=precision_recall_curve(y_test,y_lr)

closest_zero=np.argmin(np.abs(threshold))

closest_zero_p=precision[closest_zero]

closest_zero_r = recall[closest_zero]

plt.plot(precision,recall)

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.title('Precision-Recall curve with Logistic Regression')

plt.xlabel('Precision')

plt.ylabel('Recall')
from sklearn.svm import SVC
svc=SVC(gamma=1e-07,C=1e9)
svc.fit(X_train,y_train)
svc.score(X_train,y_train)
y_pred_svc=svc.predict(X_test)

svc.score(X_test,y_test)
conf_mat_svc=confusion_matrix(y_pred_svc,y_test)

labels=['Unplaced','Placed']

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = sns.heatmap(conf_mat_svc,ax=ax,annot=True,cmap='summer')

ax.xaxis.set_ticklabels(['Unplaced', 'Placed'])

ax.yaxis.set_ticklabels(['Unplaced', 'Placed'],rotation=0)

ax.set_xlabel('Predicted',size=15)

ax.set_ylabel('Actual',size=15)

plt.title('Confusion matrix SVC',size=15)

plt.figure(figsize=(10,8))
y_svc=svc.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_svc)

plt.plot(fpr,tpr,color='indianred')

plt.plot([0,1],[0,1],linestyle='--')

auc_reg=auc(fpr,tpr).round(2)

plt.title('ROC curve with AUC={}'.format(auc_reg))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')
precision,recall,threshold=precision_recall_curve(y_test,y_svc)

closest_zero=np.argmin(np.abs(threshold))

closest_zero_p=precision[closest_zero]

closest_zero_r = recall[closest_zero]

plt.plot(precision,recall)

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.title('Precision-Recall curve with SVC')

plt.xlabel('Precision')

plt.ylabel('Recall')
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

param_grid={'n_estimators':[5,7,9,10], 'max_depth':[5,7,9,10]}
rfc=RandomForestClassifier()

grid_search=GridSearchCV(rfc,param_grid,scoring='roc_auc')
grid_result=grid_search.fit(X_train,y_train)
grid_result.best_params_
grid_result.best_score_
y_pred_rfc=grid_result.predict(X_test)

grid_result.score(X_test,y_test)
conf_mat_rfc=confusion_matrix(y_pred_rfc,y_test)

labels=['Unplaced','Placed']

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = sns.heatmap(conf_mat_rfc,ax=ax,annot=True,cmap='gnuplot')

ax.xaxis.set_ticklabels(['Unplaced', 'Placed'])

ax.yaxis.set_ticklabels(['Unplaced', 'Placed'],rotation=0)

ax.set_xlabel('Predicted',size=15)

ax.set_ylabel('Actual',size=15)

plt.title('Confusion matrix RFC',size=15)

plt.figure(figsize=(10,8))
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
xgb.score(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)

xgb.score(X_test,y_test)
conf_mat_xgb=confusion_matrix(y_pred_xgb,y_test)

labels=['Unplaced','Placed']

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = sns.heatmap(conf_mat_xgb,ax=ax,annot=True,cmap='summer')

ax.xaxis.set_ticklabels(['Unplaced', 'Placed'])

ax.yaxis.set_ticklabels(['Unplaced', 'Placed'],rotation=0)

ax.set_xlabel('Predicted',size=15)

ax.set_ylabel('Actual',size=15)

plt.title('Confusion matrix XGB',size=15)

plt.figure(figsize=(10,8))