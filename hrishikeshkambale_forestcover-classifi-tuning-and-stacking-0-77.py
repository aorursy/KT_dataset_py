# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df1=pd.read_csv('../input/forest-cover-type-prediction/train.csv')
df_test1=pd.read_csv('../input/forest-cover-type-prediction/test.csv')
df_test2=pd.read_csv('../input/forest-cover-type-prediction/test3.csv')
df=df1.copy()
df_test=df_test1.copy()
df
pd.set_option('display.max_columns',None)
df.drop(columns=['Id','Cover_Type'],inplace=True)
df_test.drop(columns=['Id'],inplace=True)
df_test
X_train=df
Y_train=df1.iloc[:,-1]
X_train
df_test
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from lightgbm import LGBMClassifier

sns.set(style='white', context='notebook', palette='deep')
kfold = StratifiedKFold(n_splits=10)


random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(XGBClassifier(random_state = random_state))
classifiers.append(LGBMClassifier(random_state = random_state))

cv_results = []
for classifier in classifiers :
    score=cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=-1)
    cv_results.append(score)
    print('{} crossvalidation score:{}\n'.format(classifier,score.mean()))
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis",'XGboost','LGboost']})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X_train.values,Y_train.values,test_size=0.2)
from sklearn.metrics import accuracy_score
RFC = RandomForestClassifier(random_state=random_state)
RFC.fit(xtrain,ytrain)
ypred=RFC.predict(xtest)
score=cross_val_score(RFC,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for random forest: {}'.format(score.mean()))
print('Accuracy score for random forest: {}'.format(accuracy_score(ytest,ypred)))
RFC.get_params()
from sklearn.metrics import accuracy_score
RFC2 = RandomForestClassifier(random_state=random_state,
                             n_estimators=500,
                             max_depth=32,
                             min_samples_leaf=1,
                             criterion='entropy')
RFC2.fit(xtrain,ytrain)
ypred=RFC2.predict(xtest)
score=cross_val_score(RFC2,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for random forest: {}'.format(score.mean()))
print('Accuracy score for random forest: {}'.format(accuracy_score(ytest,ypred)))
et=ExtraTreesClassifier(random_state=random_state)
et.fit(xtrain,ytrain)
ypred=et.predict(xtest)
score=cross_val_score(et,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for extra trees classifier: {}'.format(score.mean()))
print('Accuracy score for extra trees classifier: {}'.format(accuracy_score(ytest,ypred)))

et2=ExtraTreesClassifier()
et2.get_params()
et2=ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='entropy', max_depth=38, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=500,
                     n_jobs=None, oob_score=False, random_state=0, verbose=0,
                     warm_start=False)
et2.fit(xtrain,ytrain)
ypred=et2.predict(xtest)
score=cross_val_score(et2,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for extra trees classifier: {}'.format(score.mean()))
print('Accuracy score for extra trees classifier: {}'.format(accuracy_score(ytest,ypred)))
lgb2=LGBMClassifier(random_state=random_state)
lgb2.fit(xtrain,ytrain)
ypred=lgb2.predict(xtest)
score=cross_val_score(lgb2,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for Lightgb classifier: {}'.format(score.mean()))
print('Accuracy score for Lightgb classifier: {}'.format(accuracy_score(ytest,ypred)))
lgb=LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.2, max_depth=-1,
        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
        n_estimators=200, n_jobs=4, num_leaves=63, objective=None,
        random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
lgb.fit(xtrain,ytrain)
ypred=lgb.predict(xtest)
score=cross_val_score(lgb,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for Lightgb classifier: {}'.format(score.mean()))
print('Accuracy score for Lightgb classifier: {}'.format(accuracy_score(ytest,ypred)))
vc= VotingClassifier(estimators=[('rfc', RFC2), ('extc', et2),
('lgb',lgb)], voting='soft', n_jobs=-1)
vc.fit(xtrain,ytrain)
ypred=vc.predict(xtest)
score=cross_val_score(vc,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for Lightgb classifier: {}'.format(score.mean()))
print('Accuracy score for Lightgb classifier: {}'.format(accuracy_score(ytest,ypred)))

"""
from sklearn.ensemble import StackingClassifier
estimators = [ ('rf', RFC2),
     ('et', et2)]

sc= StackingClassifier(estimators=estimators, final_estimator=lgb)
sc.fit(xtrain,ytrain)
ypred=sc.predict(xtest)
score=cross_val_score(sc,X_train,Y_train,scoring='accuracy',cv=kfold,n_jobs=-1)

# Best score
print('Crossval score for Lightgb classifier: {}'.format(score.mean()))
print('Accuracy score for Lightgb classifier: {}'.format(accuracy_score(ytest,ypred)))"""
vc.fit(X_train,Y_train)
ypred=vc.predict(df_test.values)

id=df_test1['Id']
dict={'ID':id,'Cover_Type':ypred}
dfsub=pd.DataFrame(dict)
dfsub.to_csv('./submission_ensemblevoting.csv', index=False)
"""
#ExtraTrees 
et2= ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {
 
 
 'criterion': ['gini','entropy'],
 'max_depth':[5,10,25],
 'max_features':[1,3,7],
 'max_samples': [0.2],
 'min_samples_leaf': [1,2,5],
 'min_samples_split': [2,5,7],
 'n_estimators': [100,200,300],
 }


gset = GridSearchCV(et2,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)

gset.fit(X_train,Y_train)
gset_best = gset.best_estimator_

# Best score
print(gset.best_score_)
print(gset.best_estimator_)"""

"""
# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

rf_param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_"""
"""
RFC2 = RandomForestClassifier()
rf_param_grid = {
    'bootstrap': [True],
    'max_depth': [32],
    'max_features': [2],
    'min_samples_leaf': [1],
    'min_samples_split': [6],
    'n_estimators': [300]
}


gsRFC2 = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC2.fit(X_train,Y_train)
gsRFC2.best_score_"""
pd.DataFrame(RFC.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]
pd.DataFrame(et.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(RFC,"Random Forest learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(et,"Extra trees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsRFC2,"Random Forest tuned learning curves",X_train,Y_train,cv=kfold)
#g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(lgb,"lgb tuned learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(lgb2,"Normal lgb learning curves",X_train,Y_train,cv=kfold)

#g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
#g = plot_learning_curve(vc,"voting classifier learning curves",X_train,Y_train,cv=kfold)
gset_best.fit(X_train,Y_train)
ypred=gset_best.predict(df_test.values)

id=df_test1['Id']
dict={'ID':id,'Cover_Type':ypred}
dfsub=pd.DataFrame(dict)
dfsub.to_csv('./submission_gset.csv', index=False)
