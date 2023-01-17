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
filepath='/kaggle/input/titanic/'
traindf = pd.read_csv(filepath+'train.csv')
testdf = pd.read_csv(filepath+'test.csv')
genderdf = pd.read_csv(filepath+'gender_submission.csv')
traindf.head(3)
testdf.head(3)
genderdf.head(3)
traindf.Age.plot.hist();
## Missing values

traindf.isnull().sum()
import missingno
missingno.matrix(traindf, figsize=(30,10));
dfbin = pd.DataFrame()
dfcon = pd.DataFrame()
traindf.dtypes
## Target feature 'Survived' :  1 :Survived  ,  0:Not Survived

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20,1))
sns.countplot(y='Survived', data=traindf)
print(traindf.Survived.value_counts())
dfbin['Survived']=traindf['Survived']
dfcon['Survived']=traindf['Survived']
## Feature Pclass : ticket class of passenger
## 1 : 1st class ,  2: 2nd class, 3 : 3rd class

sns.distplot(traindf.Pclass);
dfbin['Pclass']=traindf['Pclass']
dfcon['Pclass']=traindf['Pclass']
## Feature Sex : male or female

fig=plt.figure(figsize=(20,2))
sns.countplot(y='Sex', data=traindf);
print(traindf.Sex.value_counts())
dfbin['Sex'] = traindf['Sex']
dfbin['Sex'] = np.where(dfbin['Sex']=='female',1, 0)
dfcon['Sex'] = traindf['Sex']
## Sex-Survival plot

fig = plt.figure(figsize=(10,10))
sns.distplot(dfbin.loc[dfbin['Survived']==1]['Sex'], kde_kws={'label': 'Survived'});
sns.distplot(dfbin.loc[dfbin['Survived']==0]['Sex'], kde_kws={'label': 'Didnot' , 'bw':0.1});
## Feature Age
traindf.Age.isnull().sum()
def plot_count_dist(data, bindf, labelcol, targetcol, figsize=(20,5), usebindf=False):
    if usebindf:
        fig=plt.figure(figsize=figsize)
        plt.subplot(1,2,1)
        sns.countplot(y=targetcol, data=bindf);
        plt.subplot(1,2,2)
        sns.distplot(data.loc[data[labelcol]==1][targetcol], kde_kws={'Label': 'Survived', 'bw':0.1});
        sns.distplot(data.loc[data[labelcol]==0][targetcol], kde_kws={'Label': 'Did not Survive', 'bw':0.1});
    else:
        fig=plt.figure(figsize=figsize)
        plt.subplot(1,2,1)
        sns.countplot(y=targetcol, data=data);
        plt.subplot(1,2,2)
        sns.distplot(data.loc[data[labelcol]==1][targetcol], kde_kws={'Label': 'Survived', 'bw':0.1});
        sns.distplot(data.loc[data[labelcol]==0][targetcol], kde_kws={'Label': 'Did not Survive', 'bw':0.1});
## feature SibSp
dfbin['SibSp']=traindf['SibSp']
dfcon['SibSp']=traindf['SibSp']

plot_count_dist(traindf, bindf=dfbin, labelcol='Survived', targetcol='SibSp', figsize=(20,10))
# Feature : Parch
dfbin['Parch']=traindf['Parch']
dfcon['Parch']=traindf['Parch']
plot_count_dist(traindf, bindf=dfbin, labelcol='Survived', targetcol='Parch', figsize=(20,10))
# Feature : Fare
len(traindf.Fare.unique())
dfcon['Fare']=traindf['Fare']
dfbin['Fare']=pd.cut(traindf['Fare'], bins=5) ## discretized
dfbin['Fare'].head()
dfbin.Fare.value_counts()
plot_count_dist(data=traindf, bindf=dfbin, labelcol='Survived', targetcol='Fare', figsize=(20,10), usebindf=True)
# feature : Embarked
traindf.Embarked.value_counts()
sns.countplot(y='Embarked', data=traindf);
dfbin['Embarked']=traindf['Embarked']
dfcon['Embarked']=traindf['Embarked']

## drop rows with missing values for Embarked
dfcon=dfcon.dropna(subset=['Embarked'])
dfbin=dfbin.dropna(subset=['Embarked'])
len(dfcon)
dfbin.head()
## one-hot encode binned variables
onehotcols = dfbin.columns.tolist()
onehotcols.remove('Survived')
dfbinenc = pd.get_dummies(dfbin, columns=onehotcols)
dfbinenc.head()
## one hot encode the categorical columns

dfembarkedonehot = pd.get_dummies(dfcon['Embarked'], prefix='embarked')
dfsexonehot = pd.get_dummies(dfcon['Sex'], prefix='sex')
dfpclassonehot = pd.get_dummies(dfcon['Pclass'], prefix='pclass')
dfconenc = pd.concat([dfcon, dfembarkedonehot, dfsexonehot, dfpclassonehot], axis=1)
dfconenc = dfconenc.drop(['Embarked', 'Sex', 'Pclass'], axis=1)
dfconenc.head()
dfselected = dfconenc
dfselected.head()
xtrain = dfselected.drop('Survived', axis=1)
ytrain = dfselected.Survived
print(xtrain.shape)
xtrain.head()
## import libraries

from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier, Pool , cv

import datetime
import time

import warnings
warnings.filterwarnings('ignore')
def fit_ml_algo(algo, xtrain, ytrain, cv):
    
    #one pass
    model = algo.fit(xtrain, ytrain)
    acc = round(model.score(xtrain, ytrain)*100, 2)
    
    # cross-validation
    trainpred = model_selection.cross_val_predict(algo, xtrain, ytrain, cv=cv, n_jobs=-1)
    acc_cv = round(metrics.accuracy_score(ytrain, trainpred)*100 , 2)
    
    return trainpred, acc, acc_cv
## Logistic Regression

start_time = time.time()

trainpred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), xtrain, ytrain, 10)

log_time=time.time() - start_time

print(f"Accuracy : {acc_log}")
print(f"Accuracy with 10-fold CV : {acc_cv_log}")
print(f"Running Time : {datetime.timedelta(seconds=log_time)}")
## KNN
start_time = time.time()
trainpred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), xtrain, ytrain, 10)
knn_time = time.time() - start_time
print(f"Accuracy : {acc_knn}")
print(f"Accuracy with 10-Fold CV : {acc_cv_knn}")
print(f"Running time : {datetime.timedelta(seconds=knn_time)}")
## Gaussian Naive Bayes
start_time = time.time()
trainpred_gs, acc_gs, acc_cv_gs = fit_ml_algo(GaussianNB(), xtrain, ytrain, 10)
gstime = time.time() - start_time
print(f"Accuracy : {acc_gs}")
print(f"Accuracy with 10-Fold CV : {acc_cv_gs}")
print(f"Running time : {datetime.timedelta(seconds=gstime)}")
## Linear SVC

start_time = time.time()
trainpred_svc, acc_svc, acc_cv_svc = fit_ml_algo(LinearSVC(), xtrain, ytrain, 10)
linsvctime = time.time() - start_time
print(f"Accuracy : {acc_svc}")
print(f"Accuracy with 10-Fold CV : {acc_cv_svc}")
print(f"Running time :  {datetime.timedelta(seconds=linsvctime)}")
## SGD

start_time = time.time()
trainpred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(), xtrain, ytrain, 10)
sgdtime = time.time() - start_time
print(f"Accuracy : {acc_sgd}")
print(f"Accuracy with 10-Fold CV : {acc_cv_sgd}")
print(f"Running Time : {datetime.timedelta(sgdtime)}")
## Decision Tree CLassifier

start_time = time.time()
trainpred_dt , acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), xtrain, ytrain, 10)
dttime = time.time() - start_time
print(f"Accuracy : {acc_dt}")
print(f"Accuracy with 10-Fold CV : {acc_cv_dt}")
print(f"Running time : {datetime.timedelta(seconds=dttime)}")
## Gradient Boost Classifier

start_time = time.time()
trainpred_gbt , acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(), xtrain, ytrain, 10)
gbttime = time.time() - start_time
print(f"Accuracy : {acc_gbt}")
print(f"Accuracy with 10-Fold CV : {acc_cv_gbt}")
print(f"Running Time : {datetime.timedelta(seconds=gbttime)}")
## Define categorical features for catboost model

cat_features = np.where(xtrain.dtypes != np.float)[0]
cat_features
train_pool = Pool(xtrain, ytrain, cat_features)
## catboost model
catboostmodel = CatBoostClassifier(iterations=1000, custom_loss=['Accuracy'], loss_function='Logloss')

catboostmodel.fit(train_pool, plot=True)

acc_catboost = round(catboostmodel.score(xtrain, ytrain)*100 , 2)
## Perform catboost cross validation

start_time = time.time()
cvparams = catboostmodel.get_params()
cvdata = cv(train_pool, cvparams, fold_count=10, plot=True)
catbtime = time.time() - start_time

acc_cv_catboost = round(np.max(cvdata['test-Accuracy-mean'])*100 , 2)
print("------ CatBoost Metrics -------")
print(f"Accuracy : {acc_catboost}")
print(f"Accuracty for 10-Fold CV : {acc_cv_catboost}")
print(f"Runtime : {datetime.timedelta(seconds=catbtime)}")
## Regular Accuracy score

models = pd.DataFrame({'model' : ['KNN', 'Logistic Regression', 'Naive Bayes', 'Stochastic Gradient Descent', 
                                 'Linear SVC', 'Decision Tree', 'Gradient Boosting Trees', 'CatBoost'], 
                      'score' : [acc_knn, acc_log, acc_gs, acc_sgd, acc_svc, acc_dt,acc_gbt, acc_catboost]})

print("---Regular Accuracy Scores----")
models.sort_values(by='score', ascending=False)

## ---- Cross-Validation Accuracy Score -----

cv_models = pd.DataFrame({'model' : ['KNN', 'Logistic Regression', 'Naive Bayes', 'Stochastic Gradient Descent', 
                                 'Linear SVC', 'Decision Tree', 'Gradient Boosting Trees', 'CatBoost'], 
                      'score' : [acc_cv_knn, acc_cv_log, acc_cv_gs, acc_cv_sgd, acc_cv_svc, acc_cv_dt,
                                 acc_cv_gbt, acc_cv_catboost]})

print("---- Cross Validation Accuracy Scores -----")
cv_models.sort_values(by='score', ascending=False)
def feature_importance(model, data):
    fimp = pd.DataFrame({'imp':model.feature_importances_, 'col':data.columns})
    fimp = fimp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    _ = fimp.plot(kind='barh', x='col', y='imp', figsize=(20,10))
    return fimp
feature_importance(catboostmodel, xtrain)
metrics = ['Precision', 'Recall', 'F1', 'AUC']
evalmetrics = catboostmodel.eval_metrics(train_pool, metrics=metrics, plot=True)
for metric in metrics:
    print(f"{metric} : {np.mean(evalmetrics[metric])}")
xtrain.head()
testdf.head()
## one hot encode the columns in test dataframe like xtrain

test_embarked_one_hot = pd.get_dummies(testdf['Embarked'], prefix='embarked')

test_sex_one_hot = pd.get_dummies(testdf['Sex'], prefix='sex')

test_pclass_one_hot = pd.get_dummies(testdf['Pclass'], prefix='pclass')

testdf = pd.concat([testdf, test_embarked_one_hot, test_sex_one_hot, test_pclass_one_hot], axis=1)

testdf.head()
wanted_test_columns = xtrain.columns
wanted_test_columns
## make a prediction using catboostmodel on the wanted_test_columns

predictions = catboostmodel.predict(testdf[wanted_test_columns])
predictions[:10]
## submission 

submission = pd.DataFrame()
submission['PassengerId']=testdf['PassengerId']
submission['Survived']=predictions
submission.head()
len(submission)==len(testdf)
submission.to_csv('/kaggle/working/catboost_submission.csv', index=False)