#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 12:27:35 2018

@author: mollyking
"""


# coding: utf-8

# In[30]:


##############################################################
####### Costa Rican Household Poverty Level Protection #######
##############################################################

#Code By
#Andrew Green
#Ksenia Luu
#Molly King
#Zach Densmore

#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.formula.api as smf  # R-like model specification
import statsmodels.api as sm
import numpy as np
import os
import pylab

import statsmodels as sm
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#Andrews path
andrew = 'C:/Users/acgre/Desktop/Personal Work/Costa Rican Welfare/'
#Ksenia's path
ksenia = 'put path on your machine here'
#Molly's path
molly = '/Users/mollyking/Documents/Kaggle/'
#Zach's path
zach = 'put path on your machine here'


#Set working directory
#os.chdir(andrew)


#Import datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.columns)



###################################################
########### Global Functions  #####################
###################################################

### Function to make an ROC Curve
def roc(fpr, tpr):
    roc_auc = metrics.auc(fpr,tpr)
    print('Area under ROC Curve = {}'.format(roc_auc))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


### Confusion Matrix
def plot_confusion_matrix(matrix):
    plt.figure()
    np.fill_diagonal(matrix, 0)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    plt.title('Confusion Matrix')
    plt.savefig('Confusion Matrix.pdf')
    plt.savefig('Confusion Matrix')
    plt.show()


# %%

###Per the Kaggle discussions, the same household can have different values.  Examine and correct:
d={}
weird=[]
for row in train.iterrows():
    idhogar=row[1]['idhogar']
    target=row[1]['Target']
    if idhogar in d:
        if d[idhogar]!=target:
            weird.append(idhogar)
    else:
        d[idhogar]=target



len(set(weird))


# In[32]:


#Set record so the correct target value belonging to head of household is set each time
for i in set(weird):
    hhold=train[train['idhogar']==i][['idhogar', 'parentesco1', 'Target']]
    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]
    for row in hhold.iterrows():
        idx=row[0]
        if row[1]['parentesco1']!=1:
            train.at[idx, 'Target']=target



# In[33]:
            
##Initial Summary
print('')
print('----- Summary of Input Data -----')
print('')

# show the object is a DataFrame
print('Object type: ', type(train))


# show number of observations in the DataFrame
print('Number of observations: ', len(train))


print(train.describe())
print(test.describe())


print(train.shape)
print(test.shape)


#Changes pandas settings so all columns are visible
pd.set_option('max_rows',200)
pd.set_option('max_columns',200)


train.describe()


#Training set variables missing data:
#v2a1
#v18q1
#rez_esc
#public
#meaneduc
#SQBmeaned


test.describe()


#Test set variables missing data:
#v2a1
#v18q1
#rez_esc
#meaneduc
#SQBmeaned


# In[34]:


###Univariate EDA Round One - Initial Scan Prior to Imputation

print('')
print('----- Initial Univariate EDA -----')
print('')


#Histograms and Boxplots
train.hist(figsize=(20,20))
train.plot(kind= 'box', subplots=True, layout=(30,6),  figsize=(10,10))#sharex=False, sharey=False,


# %%


###Impute missing values and create missingness variables
###Opted for medians for first round, but we can test means, etc.

print('')
print('----- Value Imputation -----')
print('')


#v2a1
train['IMP_v2a1'] = train.v2a1.fillna(train.v2a1.median())
train['M_v2a1'] = train.v2a1.isnull().astype(int)


test['IMP_v2a1'] = test.v2a1.fillna(train.v2a1.median())
test['M_v2a1'] = test.v2a1.isnull().astype(int)


#v18q1
train['IMP_v18q1'] = train.v18q1.fillna(train.v18q1.median())
train['M_v18q1'] = train.v18q1.isnull().astype(int)


test['IMP_v18q1'] = test.v18q1.fillna(train.v18q1.median())
test['M_v18q1'] = test.v18q1.isnull().astype(int)


#rez_esc
train['IMP_rez_esc'] = train.rez_esc.fillna(train.rez_esc.median())
train['M_rez_esc'] = train.rez_esc.isnull().astype(int)


test['IMP_rez_esc'] = test.rez_esc.fillna(train.rez_esc.median())
test['M_rez_esc'] = test.rez_esc.isnull().astype(int)


#meaneduc
train['IMP_meaneduc'] = train.meaneduc.fillna(train.meaneduc.median())
train['M_meaneduc'] = train.meaneduc.isnull().astype(int)


test['IMP_meaneduc'] = test.meaneduc.fillna(train.meaneduc.median())
test['M_meaneduc'] = test.meaneduc.isnull().astype(int)


#SQBmeaned
train['IMP_SQBmeaned'] = train.SQBmeaned.fillna(train.SQBmeaned.median())
train['M_SQBmeaned'] = train.SQBmeaned.isnull().astype(int)


test['IMP_SQBmeaned'] = test.SQBmeaned.fillna(train.SQBmeaned.median())
test['M_SQBmeaned'] = test.SQBmeaned.isnull().astype(int)




#Find all variables with "No" instead of a 0
def no_variables(dataset,col_names):
    for col in col_names:
        if dataset[col] == 'No':
            return col


#Replace variables containing "No" values with zeros and return updated values
def replace_values(no_variables):
    dataset[col] = [0 if dataset[col]=='No' else dataset[col] for dataset[col] in dataset[col]]
    return dataset[col]


#Return functions
col_names = list(train)
#no_variables(train,col_names)
#replace_values(no_variables)

col_names = list(test)
#no_variables(test,col_names)
#replace_values(no_variables)


#Convert all variable names to lower case
train.columns = [s.lower() for s in train.columns]
test.columns = [s.lower() for s in test.columns]


# %%


###Univariate EDA Round Two - Post-Imputation Scan


print('')
print('----- Univariate EDA - Post-Imputation Scan-----')
print('')

#Histograms and Boxplots Post-Imputation
train.hist(figsize=(20,20))
train.plot(kind= 'box', subplots=True, layout=(30,6),  figsize=(10,10))#sharex=False, sharey=False,



###Response variable exploration
train['target'].hist()
train['target'].plot(kind='box')

#sm.qqplot(train['target'], line='45')
#pylab.show()


list(train)


# In[36]:


###Heatmap to explore response correlation


#Age
train2 = train[['target','r4h1','r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','r4t3','age']]
corr = train2.corr()
sns.heatmap(corr,annot=True)




# In[37]:


#Home Ownership / Lack Thereof
train2 = train[['target','tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']]
corr = train2.corr()
sns.heatmap(corr,annot=True)


#Living Conditions - can add more here
train2 = train[['target','rooms','hacdor','hacapo','overcrowding',]]
corr = train2.corr()
sns.heatmap(corr,annot=True)


#Household
train2 = train[['target','tamhog','tamviv','hhsize','parentesco1','parentesco2','parentesco3','parentesco4','hogar_nin','hogar_adul','hogar_mayor','hogar_total','dependency']]
corr = train2.corr()
sns.heatmap(corr,annot=True)


#Education
train2 = train[['target','escolari','rez_esc','hacapo','edjefe','edjefa','meaneduc']]
corr = train2.corr()
sns.heatmap(corr,annot=True)


#Possessions
train2 = train[['target','refrig','v18q1','hacapo','computer','television','mobilephone','qmobilephone']]
corr = train2.corr()
sns.heatmap(corr,annot=True)



#Location
train2 = train[['target','lugar1','lugar2','lugar3','lugar4','lugar5','lugar6','area1','area2']]
corr = train2.corr()
sns.heatmap(corr,annot=True)



#Social Conditions / Status
train2 = train[['target','estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7']]
corr = train2.corr()
sns.heatmap(corr,annot=True)



# In[46]:


train=train.fillna(train.median())
test=test.fillna(train.median())
train.replace(('yes', 'no'), (1, 0), inplace=True)
test.replace(('yes', 'no'), (1, 0), inplace=True)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
y_train_final = pd.DataFrame(train[['target','idhogar']])
mid_train = train.copy()

del mid_train['target']; del mid_train['idhogar']

mid_train.reset_index(drop=True)

new_train = pd.concat([y_train_final,mid_train],axis = 1)
print(len(new_train))
print(len(y_train_final))
names = ['SGD'#,'SVC'
         ,'Nearest Centroid'
        ,'Random Forest','Extra Trees'
        ,'Decision Tree','Gradient Booster 1.0','Gradient Booster 0.1','Gradient Booster Plain'
        ,'Multi-layer Perceptron'] 

regressors = [SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
                            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                            learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
                            n_jobs=-1, penalty='l2', power_t=0.5, random_state=None,
                            shuffle=True, tol=None, verbose=0, warm_start=False),
              #SVC(),
              NearestCentroid(metric='euclidean', shrink_threshold=None),
              RandomForestClassifier(n_estimators=10, max_depth=None,
                                     min_samples_split=2, random_state=0),
              ExtraTreesClassifier(n_estimators=10, max_depth=None,
                                   min_samples_split=2, random_state=0),
              DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                     random_state=0),
              GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=1, random_state=0),
              GradientBoostingClassifier(n_estimators=100, learning_rate=.1,
                                         max_depth=1, random_state=0),
              GradientBoostingClassifier(),
              MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1),
             LogisticRegression(multi_class='multinomial', 
                                solver='saga', 
                                verbose=1, 
                                n_jobs=-1)]
model_data = new_train.copy()


from sklearn.model_selection import KFold

N_FOLDS = 5
RANDOM_SEED = 1

cv_f1_results = np.zeros((N_FOLDS, len(names)))
cv_acc_results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):

    X_train = model_data.iloc[train_index, 3:model_data.shape[1]]
    X_test = model_data.iloc[test_index, 3:model_data.shape[1]]
    y_train = model_data.iloc[train_index, 0]
    y_test = model_data.iloc[test_index, 0]  

    index_for_method = 0  # initialize
    for name, reg_model in zip(names, regressors):

        reg_model.fit(X_train, y_train)  # fit on the train set for this fold

        # evaluate on the test set for this fold
        y_test_predict = reg_model.predict(X_test)
        fold_method_acc_result = accuracy_score(y_test, y_test_predict)
        cv_acc_results[index_for_fold, index_for_method] = fold_method_acc_result
        
        fold_method_f1_result = f1_score(y_test, y_test_predict, average='macro')
        cv_f1_results[index_for_fold, index_for_method] = fold_method_f1_result
        index_for_method += 1
  
    index_for_fold += 1

cv_acc_results_df = pd.DataFrame(cv_acc_results)
cv_acc_results_df.columns = names

cv_f1_results_df = pd.DataFrame(cv_f1_results)
cv_f1_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                        Acc Score', sep = '')     
print(cv_acc_results_df.mean())    

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                        F1 Score', sep = '')     
print(cv_f1_results_df.mean())   


train.head()
X_test = test.copy()
X_test_id = X_test['id']

del X_test['id']; del X_test['idhogar']; del X_test['target']
len(X_test)
del new_train['id']; del new_train['idhogar']; del new_train['target']
print(new_train.shape)
print(y_train_final.shape)
########### final predictions

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=1, random_state=0)
clf.fit(new_train, y_train_final['target'])  # fit on the train set for this fold
# evaluate on the test set for this fold
y_test_prob = clf.predict_proba(X_test)[:,1]
your_file = pd.concat([pd.DataFrame(X_test_id),pd.DataFrame(y_test_prob)],axis = 1, join_axes=[X_test_id.index])
your_file.columns = ['Id','Target']
your_file.index.drop
your_file.to_csv('costarica.csv', index = False)