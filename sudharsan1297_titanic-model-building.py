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
#Importing the necessary Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# Importing the dataset

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

# For easy identification of rows from the train and test, we create a feature called label and set 0 for rows from train and 1 for rows from test

train['Label']='0'

test['Label']='1'
# Concatenating train and test data.

df=pd.concat([train,test])
df.shape
# As we have the label feature, we can reset the index values.

df.index=np.arange(0,1309)
# Checking for null values

df.isna().sum()
# Cabin feature has the most null values (77%)

df['Cabin'].isna().sum()*100/len(df['Cabin'])
# Converting the feature to string type as the nan values also get converted to string values.

df['Cabin']=df['Cabin'].astype(str)
df['Cabin'].value_counts()
# Extracting the first letter of each value as it as it provides the different cabin classes in which the passengers have travelled.

list_cabin=[]

for i in df['Cabin']:

    if i!='nan':

        i=i[0]

        list_cabin.append(i)

    else:

        list_cabin.append(i)

list_cabin
df['Cabin']=list_cabin
# Converting nan to a seperate class called NA

df['Cabin']=df['Cabin'].apply(lambda x:'NA' if x=='nan' else x)
df['Cabin'].value_counts()
df['With_without_cabin']=df['Cabin'].apply(lambda x: 0 if x=='NA' else 1)
sns.countplot(df['With_without_cabin'],hue=df['Survived'])
# Imputing the missing values with the most recurring value ('S')

df['Embarked']=df['Embarked'].apply(lambda x: x if x=='S' or x=='C' or x=='Q' else 'S')
# To impute the missing values of age, we consider the most corelated features with age.

df.corr()['Age']
# Creating a feature called married from the name feature

name=list(df['Name'].values)
title=[]

for i in name:

    i=i.split(' ')[1]

    title.append(i)


married=[]

for i in title:

    if 'Mr.' in i or 'Mrs.' in i:

        married.append(1)

    else:

        married.append(0)

        
df['Married']=married
df=df.drop(['Name','Cabin','Ticket'],1)
df.head()
# Married and Pclass have the highest corelation with Age and hence we use the information from those features to impute

# the missing values.

abs(df.corr()['Age'])
age=df[['Age','Married','Pclass']]
# We have different combinations of information in the married and pclass features and we take all the combinations 

# and impute the missing values with those median values.

m0_cl1=age[(age['Married']==0) & (age['Pclass']==1)]

m0_cl2=age[(age['Married']==0) & (age['Pclass']==2)]

m0_cl3=age[(age['Married']==0) & (age['Pclass']==3)]

m1_cl1=age[(age['Married']==1) & (age['Pclass']==1)]

m1_cl2=age[(age['Married']==1) & (age['Pclass']==2)]

m1_cl3=age[(age['Married']==1) & (age['Pclass']==3)]
m0_cl1.fillna(m0_cl1['Age'].median(),inplace=True)

m0_cl2.fillna(m0_cl2['Age'].median(),inplace=True)

m0_cl3.fillna(m0_cl3['Age'].median(),inplace=True)

m1_cl1.fillna(m1_cl1['Age'].median(),inplace=True)

m1_cl2.fillna(m1_cl2['Age'].median(),inplace=True)

m1_cl3.fillna(m1_cl3['Age'].median(),inplace=True)
age_df=pd.concat([m0_cl1,m0_cl2,m0_cl3,m1_cl1,m1_cl2,m1_cl3],0)
age_df['index']=age_df.index
age_df.sort_values('index',inplace=True)
df['Age']=age_df['Age']
# As we have only 1 missing value, we can impute it with the mean value.

df['Fare'].fillna(df['Fare'].mean(),inplace=True)
# Only Survived feature has missing values and the 418 values are from the test data which we have to predict.

df.isna().sum()
# We can combine the parch and SibSp to calculate the number of family members they had on board.

df['Dependents']=df['Parch']+df['SibSp']
# Mapping the different values obtained to categorize it.

df['Dependents']=df['Dependents'].map({0:'No',1:'Few',2:'Few',3:'Few',4:'Few',5:'Few',6:'Many',7:'Many',10:'Many'})
# Dropping the Parch and SibSp features as the information has been obtained from it.

df=df.drop(['Parch','SibSp'],1)
df.head()
# As the location in which a passenger boarded does not affect whether they survived or not, we can drop it.

df=df.drop('Embarked',1)
# As the passengerID is also a feature not required for the model, we can drop it 

df=df.drop('PassengerId',1)
# Convering class to string type in order to create dummies for this feature too.

df['Pclass']=df['Pclass'].astype(str)
df=pd.get_dummies(df,drop_first=True)
# Splitting indepenndent and dependent features

X=df.drop('Survived',1)

y=df['Survived']
# Scaling the features to bring them all to one scale.

from sklearn.preprocessing import StandardScaler

X=pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)
X.head()
#Dropping the Label feature as we can split the data into train and test based on the index values. 

X=X.drop('Label_1',1)
X_train=X.iloc[:891]

X_test=X.iloc[891:]

y_train=df['Survived'].iloc[:891]

y_test=df['Survived'].iloc[891:]
# Splitting the train data into train and validation to check the performance of different models.

from sklearn.model_selection import train_test_split

train_X,X_val,train_y,y_val=train_test_split(X_train,y_train,test_size=0.25,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,f1_score
# We tune the value of C as it gives penalty to different features.

for i in [0.0001,0.001,0.1,1,10,100,1000]:

    lr=LogisticRegression(C=i).fit(train_X,train_y)

    y_pred_lr=lr.predict(X_val)

    print('For C value',i,'f1 score is: ',f1_score(y_val,y_pred_lr))
lr=LogisticRegression(C=10).fit(train_X,train_y)

y_pred_lr=lr.predict(X_val)

print('For C value 10 f1 score is: ',f1_score(y_val,y_pred_lr))
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score
# Calcutating the threshold value of where to convert the probable values as 0 and 1.

for i in range(1,11):

    y_pred2=lr.predict_proba(X_val)

    bina=binarize(y_pred2,threshold=i/10)[:,1]

    cm2=confusion_matrix(y_val,bina)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    print('f1 score: ',f1_score(y_val,bina))

    print('accuracy score: ',accuracy_score(y_val,bina))

    print('\n')
#0.4 is where we have the least misclassified values.

y_pred2=lr.predict_proba(X_val)

bina=binarize(y_pred2,threshold=0.4)[:,1]

print(confusion_matrix(y_val,bina))

print('f1_score: ',f1_score(y_val,bina))

print('accuracy_score: ',accuracy_score(y_val,bina))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier().fit(train_X,train_y)

y_pred_dt=dt.predict(X_val)

f1_score(y_val,y_pred_dt)
# Hyper Parameter Tuning

from sklearn.model_selection import GridSearchCV

dt=DecisionTreeClassifier()

param_grid = {



    'criterion': ['gini','entropy'],

    'max_depth': [10,15,20,25],

    'min_samples_split' : [5,10,15,20],

    'min_samples_leaf': [2,5,7],

    'random_state': [42,135,777],

}



rf_grid=GridSearchCV(estimator=dt,param_grid=param_grid,n_jobs=-1,return_train_score=True)



rf_grid.fit(train_X,train_y)

rf_grid.best_params_
cv_res_df=pd.DataFrame(rf_grid.cv_results_)
cv_res_df.head()
# We take the point where the test score is high and also where the difference between the train and test score is minimal

plt.figure(figsize=(20,5))

plt.plot(cv_res_df['mean_train_score'])

plt.plot(cv_res_df['mean_test_score'])

plt.xticks(np.arange(0,250,5),rotation=90)

plt.show()
cv_res_df[['mean_train_score','mean_test_score']].iloc[240:246]
pd.DataFrame(cv_res_df.iloc[240]).T
# Creating a decision tree model with the optimal hyperparameters

dt=DecisionTreeClassifier(max_depth=20,min_samples_leaf=7,min_samples_split=5,criterion='entropy',random_state=42).fit(train_X,train_y)
y_pred_dtc=dt.predict(X_val)
accuracy_score(y_val,y_pred_dtc)
f1_score(y_val,y_pred_dtc)
confusion_matrix(y_val,y_pred_dtc)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier().fit(train_X,train_y)

y_pred_rf=rf.predict(X_val)

print(accuracy_score(y_val,y_pred_rf),'\t',f1_score(y_val,y_pred_rf))
# Hyper parameter tuning

from sklearn.model_selection import GridSearchCV

rf=RandomForestClassifier()

param_grid = {

    

    'n_estimators':[10,20,30],

    'criterion': ['gini','entropy'],

    'max_depth': [10,15,20,25],

    'min_samples_split' : [5,10,15],

    'min_samples_leaf': [2,5,7],

    'random_state': [42,135,777],

    'class_weight': ['balanced' ,'balanced_subsample']

}



rf_grid=GridSearchCV(estimator=rf,param_grid=param_grid,n_jobs=-1,return_train_score=True)



rf_grid.fit(train_X,train_y)
cv_res_df=pd.DataFrame(rf_grid.cv_results_)
plt.figure(figsize=(20,5))

plt.plot(cv_res_df['mean_train_score'])

plt.plot(cv_res_df['mean_test_score'])

plt.xticks(np.arange(0,1200,50),rotation=90)

plt.show()
pd.DataFrame(cv_res_df.iloc[330])
rfc=RandomForestClassifier(class_weight='balanced',criterion='gini',max_depth=10,min_samples_leaf=2,min_samples_split=5,n_estimators=30,random_state=42).fit(train_X,train_y)
y_pred_rfc=rfc.predict(X_val)

print(accuracy_score(y_val,y_pred_rfc),'\t',f1_score(y_val,y_pred_rfc))
confusion_matrix(y_val,y_pred_rfc)
#Converting the dataset into matrix.

import xgboost as xgb

dtrain=xgb.DMatrix(train_X,train_y)

dval=xgb.DMatrix(X_val,y_val)
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }

num_round = 2

bst = xgb.train(param, dtrain, num_round)

# make prediction

preds = bst.predict(dval)
for i in range(1,11):

    bina=binarize(preds.reshape(-1,1),threshold=i/10)

    cm2=confusion_matrix(y_val,bina)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    print('f1 score: ',f1_score(y_val,bina))

    print('accuracy score: ',accuracy_score(y_val,bina))

    print('\n')
from sklearn.ensemble import GradientBoostingClassifier

gr_boost=GradientBoostingClassifier().fit(train_X,train_y)

y_pred_gr=gr_boost.predict(X_val)

print(accuracy_score(y_val,y_pred_gr),'\t',f1_score(y_val,y_pred_gr))
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4,6,8,10],

              'min_samples_leaf': [20,50,100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=5, scoring="accuracy", n_jobs= -1, verbose = 1)



gsGBC.fit(train_X,train_y)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
gsGBC.best_params_
gr_boost1=GradientBoostingClassifier(**gsGBC.best_params_).fit(train_X,train_y)

y_pred_gr1=gr_boost1.predict(X_val)

print(f1_score(y_val,y_pred_gr1),accuracy_score(y_val,y_pred_gr1))
from sklearn.ensemble import AdaBoostClassifier

ada_boost=AdaBoostClassifier().fit(train_X,train_y)

y_pred_ada=ada_boost.predict(X_val)

print(accuracy_score(y_val,y_pred_ada),'\t',f1_score(y_val,y_pred_ada))
y_pred_test_lr=lr.predict(X_test)

y_pred_test2=lr.predict_proba(X_test)
y_pred_test_lr=binarize(y_pred_test2,threshold=0.4)[:,1]
log_pred=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

log_pred['Survived']=y_pred_test_lr

log_pred.to_csv('Logistic pred.csv',index=False)
y_pred_test_dt=dt.predict(X_test)

dt_pred=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

dt_pred['Survived']=y_pred_test_dt

dt_pred.to_csv('Decision_tree_pred.csv',index=False)
y_pred_test_rfc=rfc.predict(X_test)

rfc_pred=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

rfc_pred['Survived']=y_pred_test_rfc

rfc_pred.to_csv('Random_forest_pred.csv',index=False)
dtest=xgb.DMatrix(X_test)
pred_xgb=bst.predict(dtest)
y_pred_xgb=binarize(pred_xgb.reshape(-1,1),threshold=0.4)
pred_test_xgb=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

pred_test_xgb['Survived']=y_pred_xgb
pred_test_xgb.to_csv('XGBoost_pred.csv',index=False)
y_pred_test_gr=gr_boost.predict(X_test)

pred_gr=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

pred_gr['Survived']=y_pred_test_gr

pred_gr.to_csv('GradientBoost_pred.csv',index=False)
y_pred_test_ada=ada_boost.predict(X_test)

pred_ada=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

pred_ada['Survived']=y_pred_test_ada

pred_ada.to_csv('AdaBoost_pred.csv',index=False)
from sklearn.decomposition import PCA
pca=PCA().fit(X)
pca.explained_variance_ratio_
plt.figure(figsize=(12,6))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.grid()
# From the graph, we see that 95% of the variance is explained by 6 principle components and hence we can take 6 Principle

# components.

X1=PCA(n_components=6).fit_transform(X)

X1
# Splitting train and test independent features.

X_test_pca,X_train_pca=X1[891:],X1[:891]
# We repeat the same steps for the converted data.

trai_x,val_x,trai_y,val_y=train_test_split(X_train_pca,y_train,test_size=0.3,random_state=42)
for i in [0.00001,0.0001,0.001,0.1,1,10,100,1000]:

    lr=LogisticRegression(C=i).fit(trai_x,trai_y)

    y_pred_pca_lr=lr.predict(val_x)

    print('For C value',i,'f1 score is: ',f1_score(val_y,y_pred_pca_lr))
lr=LogisticRegression(C=1).fit(trai_x,trai_y)

y_pred_pca_lr=lr.predict(val_x)

print(f1_score(val_y,y_pred_pca_lr),'\t',accuracy_score(val_y,y_pred_pca_lr))
for i in range(1,11):

    y_pred2=lr.predict_proba(val_x)

    bina=binarize(y_pred2,threshold=i/10)[:,1]

    cm2=confusion_matrix(val_y,bina)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    print('f1 score: ',f1_score(val_y,bina))

    print('accuracy score: ',accuracy_score(val_y,bina))

    print('\n')
y_pred2=lr.predict_proba(X_test_pca)

y_pred_pca_lr=binarize(y_pred2,threshold=0.4)[:,1]
sub_lr=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

sub_lr['Survived']=y_pred_pca_lr

sub_lr.to_csv('Logistic_Regression_PCA.csv',index=False)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier().fit(trai_x,trai_y)

y_pred_dt=dt.predict(val_x)

f1_score(val_y,y_pred_dt)


from sklearn.model_selection import GridSearchCV

dt=DecisionTreeClassifier()

param_grid = {



    'criterion': ['gini','entropy'],

    'max_depth': [5,10,15,20,25],

    'min_samples_split' : [5,10,15,20],

    'min_samples_leaf': [2,5,7,10],

    'random_state': [42,135,777],

}



rf_grid=GridSearchCV(estimator=dt,param_grid=param_grid,n_jobs=-1,return_train_score=True)



rf_grid.fit(trai_x,trai_y)
rf_grid.best_params_
cv_res_df=pd.DataFrame(rf_grid.cv_results_)
plt.figure(figsize=(12,6))

cv_res_df['mean_train_score'].plot()

cv_res_df['mean_test_score'].plot()

plt.xticks(np.arange(0,500,20),rotation=90)

plt.show()
cv_res_df.iloc[267]
dt=DecisionTreeClassifier(max_depth=5,min_samples_leaf=7,min_samples_split=10,criterion='entropy',random_state=42).fit(trai_x,trai_y)
y_pred_val_pca_dt=dt.predict(val_x)

print(f1_score(val_y,y_pred_val_pca_dt),accuracy_score(val_y,y_pred_val_pca_dt))
y_pred_pca_dt=dt.predict(X_test_pca)
sub_dt=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

sub_dt['Survived']=y_pred_pca_dt

sub_dt.to_csv('Decision_Tree_PCA.csv',index=False)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier().fit(trai_x,trai_y)

y_pred_rf=rf.predict(val_x)

print(accuracy_score(val_y,y_pred_rf),'\t',f1_score(val_y,y_pred_rf))
from sklearn.model_selection import GridSearchCV

rf=RandomForestClassifier()

param_grid = {

    

    'n_estimators':[10,20,30],

    'criterion': ['gini','entropy'],

    'max_depth': [10,15,20,25],

    'min_samples_split' : [5,10,15],

    'min_samples_leaf': [2,5,7],

    'random_state': [42,135,777],

    'class_weight': ['balanced' ,'balanced_subsample']

}



rf_grid=GridSearchCV(estimator=rf,param_grid=param_grid,n_jobs=-1,return_train_score=True)



rf_grid.fit(trai_x,trai_y)
rf_grid.best_params_
cv_res_df=pd.DataFrame(rf_grid.cv_results_)

cv_res_df
plt.figure(figsize=(12,6))

cv_res_df['mean_train_score'].plot()

cv_res_df['mean_test_score'].plot()
cv_res_df['diff']=cv_res_df['mean_train_score']-cv_res_df['mean_test_score']
cv_res_df[['mean_train_score','mean_test_score']][(cv_res_df['mean_test_score']>0.80) & (cv_res_df['mean_train_score']<0.90)]
cv_res_df.iloc[11]
rf=RandomForestClassifier(n_estimators=10,max_depth=10,min_samples_leaf=2,min_samples_split=10,random_state=777,criterion='gini').fit(trai_x,trai_y)

y_pred_rf=rf.predict(val_x)

print(accuracy_score(val_y,y_pred_rf),'\t',f1_score(val_y,y_pred_rf))
y_pred_pca_rf=rf.predict(X_test_pca)
sub_rf=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

sub_rf['Survived']=y_pred_pca_rf

sub_rf.to_csv('Random_Forest_PCA.csv',index=False)
dtrain_pca=xgb.DMatrix(trai_x,trai_y)

dval_pca=xgb.DMatrix(val_x)

param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }

num_round = 2

bst = xgb.train(param, dtrain_pca, num_round)

# make prediction

preds = bst.predict(dval_pca)
for i in range(1,11):

    bina=binarize(preds.reshape(-1,1),threshold=i/10)

    cm2=confusion_matrix(val_y,bina)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    print('f1 score: ',f1_score(val_y,bina))

    print('accuracy score: ',accuracy_score(val_y,bina))

    print('\n')
dtest_pca=xgb.DMatrix(X_test_pca)

preds=bst.predict(dtest_pca)

bina=binarize(preds.reshape(-1,1),threshold=0.5)
sub_xgb=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

sub_xgb['Survived']=bina

sub_xgb.to_csv('XGBoost_PCA.csv',index=False)
from sklearn.ensemble import GradientBoostingClassifier

gr_boost=GradientBoostingClassifier().fit(trai_x,trai_y)

y_pred_gr=gr_boost.predict(val_x)

print(accuracy_score(val_y,y_pred_gr),'\t',f1_score(val_y,y_pred_gr))
y_pred_pca_gr=gr_boost.predict(X_test_pca)
sub_gr=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

sub_gr['Survived']=y_pred_pca_gr

sub_gr.set_index('PassengerId')

sub_gr.to_csv('GradientBoost_PCA.csv',index=False)
ada_boost=AdaBoostClassifier().fit(trai_x,trai_y)

y_pred_val=ada_boost.predict(val_x)

print(accuracy_score(val_y,y_pred_val),'\t',f1_score(val_y,y_pred_val))
y_pred_pca_ab=ada_boost.predict(X_test_pca)
sub_ab=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])

sub_ab['Survived']=y_pred_pca_ab

sub_ab.to_csv('Adaboost_PCA.csv',index=False)