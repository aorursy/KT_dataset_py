import os
import pandas as pd

input_io_dir="../input/titanic/"

original_train_data=pd.read_csv(input_io_dir+"train.csv")
original_test_data=pd.read_csv(input_io_dir+"test.csv")
print('original_train_data',original_train_data.shape)
print('original_test_data',original_test_data.shape)
from sklearn import ensemble, model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score,train_test_split
from xgboost import XGBClassifier

input_io_dir='../input/titanic-competition-feature-engineering-1/'
# Compare different models
def PrepareDataSets():
    passengerId=pd.read_csv(input_io_dir+"passengerId.csv",header=None)
    train_features=pd.read_csv(input_io_dir+"train_features.csv",header=0)
    train_labels=pd.read_csv(input_io_dir+"train_labels.csv",header=None)
    test_features=pd.read_csv(input_io_dir+"test_features.csv",header=0)
    print('PrepareDataSets: passengerId loaded(%d)'% len(passengerId))
    print('PrepareDataSets: train_features loaded(%d)'% len(train_features))
    print('PrepareDataSets: train_labels loaded(%d)'% len(train_labels))
    print('PrepareDataSets: test_features loaded(%d)'% len(test_features))
    return passengerId,train_features,train_labels, test_features
    
def ModelSelection(clf_list,name_list,train_features,train_labels,scoring='accuracy'):
    best_score=0
    for clf, name in zip(clf_list,name_list) :
        scores = model_selection.cross_val_score(clf, train_features.values.astype(float), train_labels.values.ravel().astype(float), cv=10, scoring=scoring)  
        print("ModelSelection: Scoring  %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
        reference_score=scores.mean()+scores.std()
        if (reference_score>best_score):
            best_clf=name
            best_score=reference_score
            learning_model=clf
    print("ModelSelection: Best model - "+best_clf)
    return learning_model

def ConfigureLearningModelsForBinaryClassification():
    xgb_clf = XGBClassifier(n_estimators=100,max_depth=40, random_state=42)
    dt_clf = DecisionTreeClassifier(random_state=42)
    rf_clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    et_clf = ensemble.ExtraTreesClassifier(n_estimators=100, random_state=42)
    gb_clf = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=42)
    ada_clf = ensemble.AdaBoostClassifier(n_estimators=100, random_state=42)
    svm_clf = svm.LinearSVC(C=0.1,random_state=42)
    lg_clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=400,random_state=42)
    e_clf = ensemble.VotingClassifier(estimators=[('xgb', xgb_clf), ('dt', dt_clf),('rf',rf_clf), ('et',et_clf), ('gbc',gb_clf), ('ada',ada_clf), ('svm',svm_clf), ('lg',lg_clf)])
    clf_list = [xgb_clf, dt_clf, rf_clf, et_clf, gb_clf, ada_clf, svm_clf,lg_clf,e_clf]
    name_list = ['XGBoost', 'Decision Trees','Random Forest', 'Extra Trees', 'Gradient Boosted', 'AdaBoost', 'Support Vector Machine', 'LogisticRegression','Ensemble']
    return clf_list,name_list

passengerId,train_features,train_labels, test_features=PrepareDataSets()
clf_list,name_list=ConfigureLearningModelsForBinaryClassification()
learning_model=ModelSelection(clf_list,name_list,train_features,train_labels)

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Draw learning curve
def plot_learning_curve(ax,learning_model, title, X, y, ylim=None, cv=None, random_state=42,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    ax.set_title(title)
    if ylim is not None:
        ax.ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(learning_model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    ax.legend(loc="best")

def DrawLearningCurves(clf_list,name_list,train_features,train_labels,scoring='accuracy',cols=1,figsize=(20,20)):
    rows=len(clf_list)
    i=1
    f = plt.figure(figsize=figsize)
    for clf, name in zip(clf_list,name_list) :
        ax=f.add_subplot(rows,cols,i)
        plot_learning_curve(ax,clf,name,train_features.values.astype(float), train_labels.values.ravel().astype(float),cv=10)
        i=i+1

passengerId,train_features,train_labels, test_features=PrepareDataSets()
clf_list,name_list=ConfigureLearningModelsForBinaryClassification()
DrawLearningCurves(clf_list,name_list,train_features,train_labels,figsize=(16,60))
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fine tune a model given a param_grid
def FineTuneLearningModel(learning_model, param_grid, train_features,train_labels,scoring='accuracy'):
    grid_search = GridSearchCV(learning_model, param_grid, scoring,cv=10)
    grid_search.fit(train_features.values.astype(float),train_labels.values.ravel().astype(float))
    cvres = grid_search.cv_results_
    for mean_score,std_score, params in zip(cvres["mean_test_score"], cvres["std_test_score"],cvres["params"]):
        print('FineTuneLearningModel:',mean_score,'+-',std_score, params)
    print('FineTuneLearningModel: Best params - '+str(grid_search.best_params_))
    return grid_search.best_estimator_
# Let's run a iterative process in which we
passengerId,train_features,train_labels, test_features=PrepareDataSets()
learning_model = XGBClassifier(objective='binary:logistic')
print('FineTuneLearningModel: round 1 - booster type')
print('---------------------------------------------')
param_grid = [
    {'booster':['gbtree','gblinear'],'n_estimators': [10,30,100],'learning_rate':[0.1]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 2 - complexity params')
print('--------------------------------------------------')
param_grid = [
    {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2),'gamma':[i/10.0 for i in range(0,5)]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 3 - robustness params')
print('--------------------------------------------------')
param_grid = [
    { 'subsample':[1e-5,1e-2,0.1,0.2,0.5,0.8,1], 'colsample_bytree':[1e-5,1e-2,0.1,0.2,0.5,0.8,1]}
]     
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 4 - regularisation')
print('-----------------------------------------------')
param_grid = [
    { 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10,50],'reg_lambda':[0.1,0.5, 1, 2,5,10,50]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 5 - reduce learning rate as it prevents overfitting')
print('--------------------------------------------------------------------------------')
param_grid = [
    { 'learning_rate': [1e-5,0.01],'n_estimators': [10,100,200,500,1000]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
passengerId,train_features,train_labels, test_features=PrepareDataSets()
clf_list=[]
clf_list.append(learning_model)
name_list=['XgBoost']
DrawLearningCurves(clf_list,name_list,train_features,train_labels,figsize=(20,12))
# Train and generate predictions
def TrainModelAndGeneratePredictionsOnTestSet(learning_model,train_features,train_labels,test_features, threshold=-1):
    learning_model.fit(train_features.values.astype(float),train_labels.values.ravel().astype(float))
    if threshold==-1:
        predictions = learning_model.predict(test_features.values.astype(float))
    else:
        if hasattr(learning_model,"decision_function"):
            y_scores=learning_model.decision_function(test_features.values.astype(float))
        else:
            y_proba=learning_model.predict_proba(test_features.values.astype(float))
            y_scores=y_proba[:,1]
        predictions=(y_scores>threshold).astype(float)
    pred=pd.Series(predictions)
    # Ensure no floats go out
    return pred.apply(lambda x: 1 if x>0 else 0)
predictions=TrainModelAndGeneratePredictionsOnTestSet(learning_model,train_features,train_labels,test_features)
print('TrainModelAndGeneratePredictionsOnTestSet: predictions ready')
print('TrainModelAndGeneratePredictionsOnTestSet:Feature importances',sorted(zip(learning_model.feature_importances_,train_features.columns), reverse=True))
def GenerateOutputFile(passengerId,predictions):
    output = pd.DataFrame({ 'PassengerId': passengerId,
                            'Survived': predictions })
    output.to_csv("output.csv", index=False)

passengerId = original_test_data['PassengerId']
GenerateOutputFile(passengerId,predictions)
training_predictions = pd.DataFrame(learning_model.predict(train_features.values.astype(float)))
training_predictions.iloc[:,0]=training_predictions.iloc[:,0].astype(int)
result=original_train_data.join(training_predictions!=train_labels)

result.rename(columns={0:'Error'},inplace=True)
training_predict_error=result[result.Error==True]
training_predict_error.head()
training_predict_error.describe()