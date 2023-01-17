# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,roc_curve, confusion_matrix,make_scorer
df = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
df.shape
df.columns
df.columns = df.columns.str.lower().str.strip().str.replace(' ','_').str.replace('.','_')
df.isna().any()
df.head()
sns.set_style('darkgrid')
defaulters = df.groupby('default_payment_next_month').count()['id'].reset_index().rename(columns = {'id':'Count'})
defaulter_map = {1:'Yes',0:'No'}
defaulters['default_payment_next_month'] = defaulters['default_payment_next_month'].map(defaulter_map)
fig = plt.figure()
ax = sns.barplot(x = 'default_payment_next_month', y = 'Count', data = defaulters)
for g in ax.patches:
    current_width = g.get_width()
    new_width = current_width/2
    g.set_width(new_width)
    diff = current_width-new_width
    g.set_x(g.get_x() + diff/2)
plt.show()
gender = df.groupby(['sex','default_payment_next_month']).count()['id'].reset_index().rename(columns={'id':'counts'})
gender_map = {1:'Male',2:'Female'}
gender['sex'] = gender['sex'].map(gender_map)
gender['default_payment_next_month'] = gender['default_payment_next_month'].map(defaulter_map)
gender['percentage'] = np.round(gender['counts']/gender['counts'].sum()*100,2)
df['sex'] = df['sex'].map(gender_map)
gender
gender_dist = gender.groupby(['sex'])['counts'].sum().reset_index()
gender_dist['percentage'] = np.round(gender_dist['counts']/gender_dist['counts'].sum(),2)

fig = plt.figure(figsize = (20,7.5))
ax1 = plt.subplot(1,2,1)
ax1 = plt.pie(gender_dist['counts'],labels = gender_dist['sex'],autopct='%1.1f%%')
ax2 = plt.subplot(1,2,2)
ax2 = sns.barplot(x = 'sex', y = 'percentage', hue = 'default_payment_next_month', data = gender)
ax2.set_xlabel('Sex', fontsize = 15)
ax2.set_ylabel('Count', fontsize = 15)
ax2.tick_params(axis = 'both', labelsize = 15 )
for g in ax2.patches:
    current_width = g.get_width()
    new_width = current_width/2
    g.set_width(new_width)
    diff = current_width-new_width
    g.set_x(g.get_x() + diff/2)
plt.show()

marriage = df.groupby(['marriage','default_payment_next_month']).count()['id'].reset_index().rename(columns={'id':'Count'})
marital_status = {0:'Others', 1:'Married',2:'Single',3:'Others'}
marriage['marriage'] = marriage['marriage'].map(marital_status)
marriage['default_payment_next_month'] = marriage['default_payment_next_month'].map(defaulter_map)
marriage['percentage'] = np.round(marriage['Count']/marriage['Count'].sum()*100,2)
df['marriage'] = df['marriage'].map(marital_status)
marriage_dist = marriage.groupby(['marriage'])['Count'].sum().reset_index()
marriage_dist['percentage'] = np.round(marriage_dist['Count']/marriage_dist['Count'].sum()*100,2)
marriage
fig = plt.figure(figsize = (20,7.5))
ax1 = plt.subplot(1,2,1)
ax1 = plt.pie(marriage_dist['percentage'],labels = marriage_dist['marriage'],autopct='%1.1f%%')
ax2 = plt.subplot(1,2,2)
ax2 = sns.barplot(x = 'marriage', y = 'percentage',hue = 'default_payment_next_month', data = marriage, ci = None)
ax2.set_xlabel('Marriage',fontsize = 15)
ax2.set_ylabel('Count', fontsize = 15)
ax2.tick_params(labelsize = 15)
plt.show()

#Adding column for Age Band
df.insert(list(df.columns).index('age')+1,'age_band', pd.cut(df['age'],include_lowest = True,bins = [20,25,30,35,40,45,50,55,60,65,70,75,80],labels = ['21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70','70-75','76-80'],precision = 0 ))

df.columns
age_wise = df.groupby(['age_band']).size().reset_index().rename(columns = {0:'count'})
age_wise['percentage'] = np.round(age_wise['count']/age_wise['count'].sum()*100,2)
age_wise_defaulters = df.loc[df['default_payment_next_month']==1].groupby(['age_band','default_payment_next_month']).count()['id'].reset_index()
age_wise_defaulters.rename(columns = {'id':'count'}, inplace = True)
age_wise_defaulters['count'].fillna(0,inplace = True)
age_wise_defaulters['percentage'] = np.round(age_wise_defaulters['count']/age_wise_defaulters['count'].sum()*100,2)
age_wise
age_wise_defaulters
fig = plt.figure(figsize = (20,7))
ax1 = plt.subplot(1,2,1)
ax1 = sns.barplot(x = age_wise['age_band'], y = age_wise['percentage'])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)
ax1.set_xlabel('Age Group', fontsize = 15)
ax1.set_ylabel('Percentage', fontsize = 15)
ax2 = plt.subplot(1,2,2)
ax2 = sns.barplot(x = age_wise_defaulters['age_band'], y = age_wise_defaulters['percentage'])
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 45)
ax2.set_xlabel('Age Group', fontsize = 15)
ax2.set_ylabel('Percentage', fontsize = 15)
ax2.tick_params(labelsize = 10)
plt.show()
#EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
df.loc[(df['education']==5)| (df['education']==6) | (df['education']==0), 'education'] = 4
education_matrix = {1:'Graduate School', 2:'University', 3:'High School', 4:'Others'}
education = df.groupby(['education']).count()['id'].reset_index().rename(columns = {'id':'count'})
education['education'] = education['education'].map(education_matrix)
education['Percentage'] = np.round(education['count']/education['count'].sum()*100,2)
fig = plt.figure(figsize = (10,10))
sizes = education['Percentage']
labels = education['education']
ax = plt.pie(sizes, labels = labels,autopct='%1.1f%%')
plt.show()
df['education'] = df['education'].map(education_matrix)
df.groupby(['age_band'])['limit_bal'].median().reset_index()
fig = plt.figure(figsize = (20,10))
ax = sns.boxplot(y = 'age_band', x = 'limit_bal', data = df, orient = 'h', )
ax.set_xlabel('Limit Balance',fontsize = 15)
ax.set_ylabel('Age Band', fontsize = 15)
ax = plt.locator_params(axis='x', nbins=20)
plt.show()
fig = plt.figure(figsize = (20,10))
g = sns.FacetGrid(df, col = 'default_payment_next_month',row = 'marriage',hue = 'sex',legend_out = True, margin_titles = True, )
g.map(plt.hist, 'age', bins = 25,alpha=0.5)
g.add_legend()
plt.show()
fig = plt.figure(figsize = (20,10))
g = sns.FacetGrid(df, col = 'default_payment_next_month',row = 'education',hue = 'sex',legend_out = True, margin_titles = True, )
g.map(plt.hist, 'age',bins=25,alpha=0.5)
g.add_legend()
plt.show()
df.loc[df['default_payment_next_month']==1,['bill_amt1','pay_0','bill_amt2','pay_2','bill_amt3','pay_3','bill_amt4','pay_4','bill_amt5','pay_5','bill_amt6','pay_6','default_payment_next_month']].sample(50)
df.loc[(df['default_payment_next_month']==0),['bill_amt1','pay_0','bill_amt2','pay_2','bill_amt3','pay_3','bill_amt4','pay_4','bill_amt5','pay_5','bill_amt6','pay_6','default_payment_next_month']].sample(50)
import re
num = []
for var in df.columns[df.columns.str.contains(r'pay_[0-9]')]:
    num.append(var[-1])
print('Non Defaulters')
for number in num:
    print('pay_',number, ' :',sorted(list(df.loc[df['default_payment_next_month']==0,'pay_'+number].unique())))
print('\n')
print('Defaulters')
for number in num:
    print('pay_',number, ' :',sorted(list(df.loc[df['default_payment_next_month']==0,'pay_'+number].unique())))
for number in num:
    df.loc[(df['pay_'+number]==-1),'pay_' + number]=1
    df.loc[(df['pay_'+number]==-2),'pay_' + number]=2
print('Non Defaulters')
for number in num:
    print('pay_',number, ' :',sorted(list(df.loc[df['default_payment_next_month']==0,'pay_'+number].unique())))
print('\n')
print('Defaulters')
for number in num:
    print('pay_',number, ' :',sorted(list(df.loc[df['default_payment_next_month']==0,'pay_'+number].unique())))
data = df.copy()
data.drop('id',axis = 1, inplace = True)
data.columns
data.drop('age_band',axis = 1, inplace = True)
for number in num:
    data['pay_'+number] = df['pay_'+number].astype('object')
data = pd.get_dummies(data)
x_train,x_test,y_train,y_test = train_test_split(data.drop('default_payment_next_month',axis = 1),data['default_payment_next_month'], test_size = 0.25, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
param_grid = {'criterion':['gini','entropy'],
              'max_leaf_nodes':[10,20,30],
              'min_samples_leaf':[2,5,10,20],
              'max_depth':[5,10,15]}
scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score)
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state = 42),param_grid,cv=5, n_jobs = -1, scoring = scorers, refit = 'f1_score' )
dt_grid.fit(X_train,y_train)
dt_clf = dt_grid.best_estimator_
dt_clf.fit(X_train,y_train)
dt_pred = dt_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = dt_pred)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = dt_pred),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = dt_pred),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = dt_pred),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = dt_pred),2)))
param_grid = {'criterion':['gini','entropy'],
              'n_estimators':[50,100,150],
              'max_leaf_nodes':[10,20,30],
              'min_samples_leaf':[2,5,10,20],
              'max_depth':[5,10,15]}
scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score)
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state = 42),param_grid,cv=5, n_jobs = -1, scoring = scorers, refit = 'f1_score' )
rf_grid.fit(X_train,y_train)
rf_grid.best_estimator_
rf_clf = rf_grid.best_estimator_
rf_clf.fit(X_train,y_train)
rf_pred = rf_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = rf_pred)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = rf_pred),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = rf_pred),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = rf_pred),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = rf_pred),2)))
param_grid = {'C' : [0.15,0.5,1],
              'kernel' : 'rbf',
              'gamma': [0.1,1,10]}
scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score)
}
#SVM_grid = GridSearchCV(SVC(random_state = 42),param_grid,cv=5, n_jobs = -1, scoring = scorers, refit = 'f1_score' )
#SVM_grid.fit(X_train,y_train)

SVM_clf = SVC(kernel = 'rbf', C= 0.35, gamma = 0.1)
SVM_clf.fit(X_train,y_train)
SVM_pred = SVM_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = SVM_pred)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = SVM_pred),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = SVM_pred),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = SVM_pred),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = SVM_pred),2)))
ada_clf = AdaBoostClassifier(random_state = 42)
ada_clf.fit(X_train,y_train)
ada_pred = ada_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = ada_pred)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = ada_pred),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = ada_pred),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = ada_pred),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = ada_pred),2)))
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train,y_train)
xgb_pred = xgb_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = xgb_pred)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = xgb_pred),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = xgb_pred),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = xgb_pred),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = xgb_pred),2)))
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
sm = SMOTE(random_state = 42)
X_SMOTE, Y_SMOTE = sm.fit_sample(X_train,y_train)
dt_clf_SMOTE = dt_grid.best_estimator_
dt_clf_SMOTE.fit(X_SMOTE,Y_SMOTE)
dt_pred_SMOTE = dt_clf_SMOTE.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = dt_pred_SMOTE)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = dt_pred_SMOTE),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = dt_pred_SMOTE),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = dt_pred_SMOTE),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = dt_pred_SMOTE),2)))
rf_clf_SMOTE = rf_grid.best_estimator_
rf_clf_SMOTE.fit(X_SMOTE,Y_SMOTE)
rf_pred_SMOTE = rf_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = rf_pred_SMOTE)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = rf_pred_SMOTE),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = rf_pred_SMOTE),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = rf_pred_SMOTE),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = rf_pred_SMOTE),2)))
SVM_clf_SMOTE = SVC(kernel = 'rbf', C= 0.35, gamma = 0.1)
SVM_clf_SMOTE.fit(X_SMOTE,Y_SMOTE)
SVM_pred_SMOTE = SVM_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = SVM_pred_SMOTE)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = SVM_pred_SMOTE),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = SVM_pred_SMOTE),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = SVM_pred_SMOTE),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = SVM_pred_SMOTE),2)))
ada_clf_SMOTE = AdaBoostClassifier(random_state = 42)
ada_clf_SMOTE.fit(X_SMOTE,Y_SMOTE)
ada_pred_SMOTE = ada_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = ada_pred_SMOTE)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = ada_pred_SMOTE),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = ada_pred_SMOTE),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = ada_pred_SMOTE),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = ada_pred_SMOTE),2)))
xgb_clf_SMOTE = XGBClassifier()
xgb_clf_SMOTE.fit(X_SMOTE,Y_SMOTE)
xgb_pred_SMOTE = xgb_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = xgb_pred_SMOTE)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = xgb_pred_SMOTE),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = xgb_pred_SMOTE),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = xgb_pred_SMOTE),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = xgb_pred_SMOTE),2)))
sm2 = SMOTE(sampling_strategy = 0.45, random_state = 42)
under = RandomUnderSampler(sampling_strategy = 0.5, random_state = 42)
X_SMOTE2, Y_SMOTE2 = sm2.fit_resample(X_train,y_train)
X_resample,Y_resample = under.fit_resample(X_SMOTE2, Y_SMOTE2)

dt_clf_resample = dt_grid.best_estimator_
dt_clf_resample.fit(X_resample,Y_resample)
dt_pred_resample = dt_clf_SMOTE.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = dt_pred_resample)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = dt_pred_resample),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = dt_pred_resample),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = dt_pred_resample),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = dt_pred_resample),2)))
rf_clf_resample = rf_grid.best_estimator_
rf_clf_resample.fit(X_SMOTE,Y_SMOTE)
rf_pred_resample = rf_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = rf_pred_resample)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = rf_pred_resample),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = rf_pred_resample),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = rf_pred_resample),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = rf_pred_resample),2)))
SVM_clf_resample = SVC(kernel = 'rbf', C= 0.35, gamma = 0.1)
SVM_clf_resample.fit(X_resample,Y_resample)
SVM_pred_resample = SVM_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = SVM_pred_resample)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = SVM_pred_resample),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = SVM_pred_resample),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = SVM_pred_resample),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = SVM_pred_resample),2)))
ada_clf_resample = AdaBoostClassifier(random_state = 42)
ada_clf_resample.fit(X_resample,Y_resample)
ada_pred_resample = ada_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = ada_pred_resample)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = ada_pred_resample),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = ada_pred_resample),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = ada_pred_resample),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = ada_pred_resample),2)))
xgb_clf_resample = XGBClassifier()
xgb_clf_resample.fit(X_resample,Y_resample)
xgb_resample = xgb_clf.predict(X_test)
print('Accuracy:'+  str(np.round(accuracy_score(y_true = y_test,y_pred = xgb_resample)*100,2)))
print('F1 Score:'+  str(np.round(f1_score(y_true = y_test,y_pred = xgb_resample),2)))
print('Precision:'+  str(np.round(precision_score(y_true = y_test,y_pred = xgb_resample),2)))
print('Recall:'+  str(np.round(recall_score(y_true = y_test,y_pred = xgb_resample),2)))
print('ROC_AUC:'+  str(np.round(roc_auc_score(y_true = y_test,y_score = xgb_resample),2)))
