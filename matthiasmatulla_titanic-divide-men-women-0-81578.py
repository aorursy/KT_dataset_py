import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RandomizedSearchCV

import lightgbm as lgb
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

RS=81
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
data=pd.concat([train, test], sort = True).reset_index(drop=True)
data.info()
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        return 'Mr'
    elif title in ['Countess', 'Mme', 'Lady', 'Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title']=data.apply(replace_titles, axis=1)
data.groupby(['Title', 'Pclass'])['Age'].median()
data['Age']=data.groupby(['Title', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Family_Size']=data['SibSp']+data['Parch']
data['Alone']=[1 if x==0 else 0 for x in data['SibSp']+data['Parch']]
data['Surname'] =data.Name.str.extract(r'([A-Za-z]+),', expand=False)
df=data[data.duplicated(['Ticket','Surname'], keep=False)]

dfg=df.groupby(['Ticket','Surname'])[['Survived']].max().fillna(0.5)
data=pd.merge(data, dfg.rename(columns={'Survived':'FamSurvived'}), how='left', on=['Ticket','Surname'])
data['FamSurvived'].fillna(0.5, inplace=True)
data['Age_T']=PowerTransformer().fit_transform(data[['Age']])
data['Fare_T']=PowerTransformer().fit_transform(data[['Fare']])
data = pd.get_dummies(data, columns=['Title'], drop_first=True)
train = data[:len(train)]
test = data[len(train):].drop(['Survived'],axis=1)
data.groupby(['Sex', 'Pclass'])['Survived'].mean()
train_1=train.loc[(train.Sex=='male')]
test_1=test.loc[(test.Sex=='male')]

train_2=train.loc[((train.Pclass<=2) & (train.Sex=='female'))]
test_2=test.loc[((test.Pclass<=2) & (test.Sex=='female'))]

train_3=train.loc[((train.Pclass>2) & (train.Sex=='female'))]
test_3=test.loc[((test.Pclass>2) & (test.Sex=='female'))]
col_final=['Age_T', 'Fare_T', 'Alone', 'Family_Size', 'FamSurvived', 'Title_Miss', 'Title_Mr', 'Title_Mrs']
param_lgb ={'n_estimators': [100, 500, 1000, 2000],
             'max_depth':[1,2,3,4,5],
             'num_leaves': [2,4,6,8,10], 
             'min_child_samples ': [2,5,10,15,20], 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample':[0.4,0.5,0.6,0.7,0.8,0.9,1],
             'colsample_bytree':[0.4,0.5,0.6,0.7,0.8,0.9,1],
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


param_log={'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 3, 5, 7, 10, 15, 20, 25, 30, 50], 
                 'penalty':['l1', 'l2', 'elasticnet', 'none'],
                 'class_weight': [1, 3, 10],
                 'max_iter': [10000]}


param_rf=  {'n_estimators': [10 ,25,50,100,125,250,500,1000],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [1,2,3,4,5,10],
               'min_samples_split': [2, 5, 10, 15, 25],
               'min_samples_leaf': [1, 2, 5, 10]}

paramsStack= {'final_estimator__multi_class':['ovr'],
              'final_estimator__max_iter':[10000], 
                     'final_estimator__solver':['saga'],
                     'final_estimator__penalty':['l1', 'l2', 'elasticnet', 'none'],
                     'final_estimator__l1_ratio':[0.5],
                     'final_estimator__C':np.linspace(start = 0.0001, stop = 10, num = 100)}

searchParamsModels={'n_iter':50,'scoring':'accuracy','cv':10,'refit':True,'random_state':RS,'verbose':0,'n_jobs':-1}

clf_lgb=lgb.LGBMClassifier(random_state=RS, silent=True, metric='None')
clf_log=LogisticRegression(random_state = RS, solver='saga', l1_ratio=0.5)
clf_rf=RandomForestClassifier(random_state = RS)
stackM=LogisticRegression(random_state=RS)

l_param=[param_log,param_rf,param_lgb]
l_clf=[clf_log,clf_rf,clf_lgb]
def tune_stack_predict(X_train,y_train,X_test,listModels,listSearchParamM,stackModel,searchParamsModels,listSearchParamS):
    models_local=[]
    X_test_c=X_test.copy()
    
    for clf, param in zip(listModels,listSearchParamM):
        #search model param
        rs_clf=RandomizedSearchCV(estimator=clf, param_distributions=param, **searchParamsModels)
        rs_clf.fit(X_train, y_train)
        
        #model name first 5 char
        rs_clf_name=type(clf).__name__[0:5]
        
        #pred test 
        y_pred=rs_clf.best_estimator_.predict(X_test)
        
        #add pred to test
        X_test_c[rs_clf_name]=np.asarray(y_pred)
        
        #add best model to list for stack and return
        models_local.append([rs_clf_name,rs_clf.best_estimator_])
        
     
    #init stack
    stack_clf=StackingClassifier(estimators=models_local, final_estimator=stackModel,cv=10, verbose=0)
    
    #search stack param
    rs_stack=RandomizedSearchCV(estimator=stack_clf, param_distributions=listSearchParamS, **searchParamsModels)
    rs_stack.fit(X_train,y_train)
    
    #stack name first 5 char
    rs_stack_name=type(stack_clf).__name__[0:5]
  
    #add stack pred to test
    y_pred=rs_stack.best_estimator_.predict(X_test)
    X_test_c[rs_stack_name]=np.asarray(y_pred)

    #add stack to list for return
    models_local.append([rs_stack_name,rs_stack.best_estimator_])
    
    return X_test_c, models_local
#x1,m1=tune_stack_predict(train_1[col_final], train_1['Survived'],test_1[col_final],l_clf,l_param,stackM,searchParamsModels,paramsStack)
#x3,m3=tune_stack_predict(train_3[col_final], train_3['Survived'],test_3[col_final],l_clf,l_param,stackM,searchParamsModels,paramsStack)

#m1[3][1].get_params(False)
#m3[3][1].get_params(False)

param_meta1={'cv': 10,
 'estimators': [['Logis',
   LogisticRegression(C=0.4, class_weight=10, l1_ratio=0.5, max_iter=10000,
                      penalty='l1', random_state=81, solver='saga')],
  ['Rando',
   RandomForestClassifier(max_depth=3, min_samples_leaf=5, min_samples_split=5,
                          n_estimators=25, random_state=81)],
  ['LGBMC',
   LGBMClassifier(colsample_bytree=1, max_depth=3, metric='None',
                  min_child_samples =10, min_child_weight=10.0, n_estimators=500,
                  n_jobs=4, num_leaves=6, random_state=81, reg_alpha=2,
                  reg_lambda=50, subsample=0.6)]],
 'final_estimator': LogisticRegression(C=3.3334, l1_ratio=0.5, max_iter=10000, multi_class='ovr',
                    random_state=81, solver='saga'),
 'n_jobs': None,
 'passthrough': False,
 'stack_method': 'auto',
 'verbose': 0}

param_meta3={'cv': 10,
 'estimators': [['Logis',
   LogisticRegression(C=30, class_weight=1, l1_ratio=0.5, max_iter=10000,
                      penalty='none', random_state=81, solver='saga')],
  ['Rando',
   RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=2,
                          min_samples_split=15, n_estimators=1000,
                          random_state=81)],
  ['LGBMC',
   LGBMClassifier(colsample_bytree=0.5, max_depth=1, metric='None',
                  min_child_samples =15, min_child_weight=0.01, n_estimators=1000,
                  n_jobs=4, num_leaves=6, random_state=81, reg_alpha=0,
                  reg_lambda=50, subsample=0.6)]],
 'final_estimator': LogisticRegression(C=3.3334, l1_ratio=0.5, max_iter=10000, multi_class='ovr',
                    random_state=81, solver='saga'),
 'n_jobs': None,
 'passthrough': False,
 'stack_method': 'auto',
 'verbose': 0}
stack_clf1=StackingClassifier(**param_meta1)
stack_clf3=StackingClassifier(**param_meta3)
stack_clf1.fit(train_1[col_final],train_1['Survived'])
stack_clf3.fit(train_3[col_final],train_3['Survived'])
y_pred_1=stack_clf1.predict(test_1[col_final])
y_pred_3=stack_clf3.predict(test_3[col_final])
y_pred_1=pd.DataFrame({'PassengerId':test_1['PassengerId'],'Survived':y_pred_1 }, dtype=int)
y_pred_2=pd.DataFrame({'PassengerId':test_2['PassengerId'],'Survived':1 }, dtype=int)
y_pred_3=pd.DataFrame({'PassengerId':test_3['PassengerId'],'Survived':y_pred_3 }, dtype=int)
y_sub=pd.concat([y_pred_1,y_pred_2,y_pred_3], axis=0).sort_values('PassengerId')
y_sub.to_csv("sub.csv",index=False)