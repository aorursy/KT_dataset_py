import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Stats and other tools
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


#Models we will test and try
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

cv_k_global = 10 #the amount of f_folds to be used in all CV

datasets=['1','2']
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    logmodel = LogisticRegression()
    param_grid = [{
        'C': [10**x for x in range(-3,4)],
        'penalty':['l1'],
        'solver' :['liblinear','saga']},{
        'C': [10**x for x in range(-3,4)],
        'penalty':['l2'],
        'solver':['newton-cg','lbfgs','sag']}]
    gscv=GridSearchCV(logmodel,param_grid,scoring='accuracy',cv=cv_k_global,n_jobs=-1,verbose=1,refit=False)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_LR_'+dataset+'.csv')
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    estimators = [('AddPoly',PolynomialFeatures()),('LR',LogisticRegression())]
    quartic_log_reg = Pipeline(estimators)
    param_grid = [{
        'AddPoly__degree':[2,3],
        'LR__C': [10**x for x in range(-3,4)],
        'LR__penalty':['l1'],
        'LR__solver' :['liblinear','saga']
        },{
        'AddPoly__degree':[2,3],
        'LR__C': [10**x for x in range(-3,4)],
        'LR__penalty':['l2'],
        'LR__solver' :['newton-cg','lbfgs','sag']
        }]
    gscv=GridSearchCV(quartic_log_reg,param_grid,scoring='accuracy',cv=cv_k_global,verbose=1,n_jobs=-1,refit=False)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_LR_X_'+dataset+'.csv')
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    classif_ABC = AdaBoostClassifier()
    param_grid ={'n_estimators':[10*x for x in range(1,21)],
             'learning_rate':[10**x for x in range(-4,5)],
             'algorithm':['SAMME','SAMME.R']}
    gscv=GridSearchCV(classif_ABC,param_grid,scoring='accuracy',cv=cv_k_global,n_jobs=-1,verbose=1,refit=False)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_ABC_'+dataset+'.csv')
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    classif_GBC = GradientBoostingClassifier()
    param_grid ={'loss' : ['deviance', 'exponential'],
             'learning_rate':[10**x for x in range(-4,5)],
             'criterion':['friedman_mse','mse'],
             'n_estimators':[10*x for x in range(1,21)],
             'subsample':[x*0.1 for x in range(1,11)],
             'max_features':['auto','log2','sqrt',None]}
    gscv=GridSearchCV(classif_GBC,param_grid,scoring='accuracy',cv=cv_k_global,n_jobs=-1,verbose=1,refit=False)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_GBC_'+dataset+'.csv')
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    rfc = RandomForestClassifier()
    param_grid ={
        'n_estimators':[10*x for x in range(1,21)],
        'criterion':['gini','entropy'],
        'max_features':['auto','log2','sqrt',None]}
    gscv=GridSearchCV(rfc,param_grid,scoring='accuracy',cv=cv_k_global,verbose=1,n_jobs=-1,refit=False)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_RFC_'+dataset+'.csv')
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    knn = KNeighborsClassifier()
    param_grid = [{
        'n_neighbors':[2*x+1 for x in range(1,51)],
        'algorithm':[ 'ball_tree', 'kd_tree','brute'],
        'p':[1,2],
        'weights':['uniform','distance'],
        'leaf_size':[2*x+1 for x in range(1,51)]}]
    gscv=GridSearchCV(knn,param_grid,scoring='accuracy',cv=cv_k_global,verbose=1,n_jobs=-1,refit=False)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_KNN_'+dataset+'.csv')
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    svm_class = svm.SVC()
    param_grid =[
        {'C':[10**x for x in range(-3,4)],
        'kernel':['poly'],
        'degree':[2,3,4,5],
        'decision_function_shape':['ovo','ovr']
        },
         {'C':[10**x for x in range(-3,4)],
        'kernel':['rbf','linear','sigmoid'],
        'decision_function_shape':['ovo','ovr']
         }
        ]
    gscv=GridSearchCV(svm_class,param_grid,scoring='accuracy',cv=cv_k_global,verbose=1,n_jobs=-1,refit=False)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_SVM_'+dataset+'.csv')
