import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from xgboost.sklearn import XGBClassifier

from sklearn import cross_validation, metrics   #Additional scklearn functions

from sklearn.grid_search import GridSearchCV   #Perforing grid search



%matplotlib inline
df=pd.read_csv('../input/HR_comma_sep.csv')
df =pd.DataFrame(df)

df.head()
fig,ax= plt.subplots(figsize=(9,7))

sns.heatmap(df.corr(),annot=True)
fig,ax= plt.subplots(figsize=(9,5));

sns.countplot(x='number_project',data=df,hue='left');

plt.title('Number of Projects done by an employee who has left or Not');
fig,ax= plt.subplots(figsize=(7,4));

sns.countplot(x='Work_accident',data=df,hue='left');

plt.title('Work accidents with an employee who has left or Not');
fig,ax= plt.subplots(figsize=(9,5))

sns.countplot(x='time_spend_company',data=df,hue='left');

plt.title('Time spend by an employee who has left or Not');
fig,ax= plt.subplots(figsize=(9,5))

sns.countplot(x='sales',data=df,hue='left');

plt.title('Number of Projects done by an employee who has left or Not');
fig,ax= plt.subplots(figsize=(9,5))

sns.countplot(x='promotion_last_5years',data=df,hue='left');

plt.title('Number of Projects done by an employee who has left or Not');
df[(df.promotion_last_5years==1)].left.value_counts()
fig,ax= plt.subplots(figsize=(9,5))

sns.countplot(x='salary',data=df,hue='left');

plt.title('Number of Projects done by an employee who has left or Not');
df[(df.salary=='high')].left.value_counts()
fig,ax= plt.subplots(figsize=(7,5))

sns.barplot(x='left',y='satisfaction_level',data=df);

plt.title('Number of Projects done by an employee who has left or Not');
df_left = df[df.left == 1].reset_index(drop=True)

df_not_left = df[df.left == 0].reset_index(drop=True)
fig,(ax1,ax2,ax3) =plt.subplots(1,3,figsize=(16,7))

sns.distplot(df.satisfaction_level,ax=ax1)# of all employee

sns.distplot(df_left.satisfaction_level,ax=ax2)# of employee who left there job

sns.distplot(df_not_left.satisfaction_level,ax=ax3)# of employee who are doing ther job

plt.title('Of employee didnt left there job');
fig,(ax1,ax2,ax3) =plt.subplots(1,3,figsize=(16,7))

sns.distplot(df.last_evaluation,ax=ax1)# of all employee

sns.distplot(df_left.last_evaluation,ax=ax2)# of employee who left there job

sns.distplot(df_not_left.last_evaluation,ax=ax3)# of employee who are doing ther job

plt.title('Of employee didnt left there job');
fig,(ax1,ax2,ax3) =plt.subplots(1,3,figsize=(16,7))

sns.distplot(df.average_montly_hours,ax=ax1)# of all employee

sns.distplot(df_left.average_montly_hours,ax=ax2)# of employee who left there job

sns.distplot(df_not_left.average_montly_hours,ax=ax3)# of employee who are doing ther job

plt.title('Of employee didnt left there job');
f, (ax1,ax2) = plt.subplots(1,2,figsize=(15, 8))

ax.set_aspect("equal")



# Draw the two density plots

sns.kdeplot(df_left.last_evaluation, df_left.satisfaction_level,

                 cmap="Reds", shade=True, shade_lowest=False,ax=ax1)

sns.kdeplot(df_not_left.last_evaluation, df_not_left.satisfaction_level,

                 cmap="Blues", shade=True, shade_lowest=False,ax=ax2)

fig,ax= plt.subplots(figsize=(12,5))

sns.violinplot(y=df.satisfaction_level,x=df.salary,hue=df.left)
fig,ax= plt.subplots(figsize=(12,5))

sns.violinplot(y=df.satisfaction_level,x=df.promotion_last_5years,hue=df.left)
df.count()
df.describe()
bins=[90,150,200,250,350]

df['time']=np.digitize(df.average_montly_hours,bins)
df.head()
var=['sales','salary']

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in var:

    df[i]=le.fit_transform(df[i])
from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier()

col=df.columns.drop('left')

clf.fit(df[col],df['left'])

imp = clf.feature_importances_

lev = pd.DataFrame(col)

lev['imp']=imp

lev.sort_values('imp',ascending=False)
df.columns
from sklearn.cross_validation import train_test_split

train,test= train_test_split(df,test_size=0.2)



param_test1 = {

 'max_depth':np.arange(3,10,2),

 'min_child_weight':np.arange(1,6,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train[col],train['left'])

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


param_test6 = {

 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.6, n_estimators=100, max_depth=9,

 min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(train[col],train['left'])

gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
param_test4 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,

 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(train[col],train['left'])

gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_




clf=XGBClassifier(learning_rate=0.6,n_estimators=1000,max_depth=9,reg_alpha=1,min_child_weight=1)
col=['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 'time_spend_company','sales','time','salary']



clf.fit(train[col],train['left'])



pred =clf.predict(test[col])
import sklearn

ar=sklearn.metrics.confusion_matrix(test.left, pred)

ar
(ar[0][0]+ar[1][1])/(ar[0][0]+ar[1][1]+ar[1][0]+ar[0][1])