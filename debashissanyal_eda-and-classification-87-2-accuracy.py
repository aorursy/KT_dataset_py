import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sns.set() ##set defaults
%matplotlib inline
data = pd.read_csv("../input/adult.csv")
data.shape
data.head()
data.info()
data['workclass'].value_counts()
data['income'].value_counts()    
data['income'] = data['income'].apply(lambda inc: 0 if inc == "<=50K" else 1) # Binary encoding of the target variable
plt.figure(figsize=(10,5))

sns.countplot(data['income'])
plt.figure(figsize=(14,6))

sns.countplot(data['marital.status'])
plt.figure(figsize=(15,6))

ax=sns.barplot(x='marital.status',y='income',data=data,hue='sex')

ax.set(ylabel='Fraction of people with income > $50k')

data['marital.status'].value_counts()
plt.figure(figsize=(12,6))

sns.countplot(data['workclass'])
plt.figure(figsize=(12,6))

ax=sns.barplot('workclass', y='income', data=data, hue='sex')

ax.set(ylabel='Fraction of people with income > $50k')
plt.figure(figsize=(10,8))  

sns.heatmap(data.corr(),cmap='Accent',annot=True)

#data.corr()

plt.title('Heatmap showing correlations between numerical data')
plt.figure(figsize=(12,6))

sns.boxplot(x="income", y="age", data=data, hue='sex')

#data[data['income']==0]['age'].mean()
norm_fnl = (data["fnlwgt"] - data['fnlwgt'].mean())/data['fnlwgt'].std()

plt.figure(figsize=(8,6))

sns.boxplot(x="income", y=norm_fnl, data=data)
data[norm_fnl>2].shape
plt.figure(figsize=(10,5))

ax = sns.barplot(x='sex',y='income',data=data)

ax.set(ylabel='Fraction of people with income > $50k')
plt.figure(figsize=(12,6))

sns.boxplot(x='income',y ='hours.per.week', hue='sex',data=data)
plt.figure(figsize=(15,6))

ax = sns.barplot(x='marital.status',y='hours.per.week',data=data,hue='sex')

ax.set(ylabel='mean hours per week')
plt.figure(figsize=(10,6))

ax = sns.barplot(x='income', y='education.num',hue='sex', data=data)

ax.set(ylabel='Mean education')
print(data['race'].value_counts())

plt.figure(figsize=(12,6))

ax=sns.barplot(x='race',y='income',data=data)

ax.set(ylabel='Fraction of people with income > $50k')
plt.figure(figsize=(12,6))

sns.jointplot(x=data['capital.gain'], y=data['capital.loss'])

#print(data[((data['capital.gain']!=0) & (data['capital.loss']!=0))].shape)
plt.figure(figsize=(12,8))

sns.distplot(data[(data['capital.gain']!=0)]['capital.gain'],kde=False, rug=True)
plt.figure(figsize=(12,8))

sns.distplot(data[(data['capital.loss']!=0)]['capital.loss'], kde=False,rug=True)
plt.figure(figsize=(20,6))

ax=sns.barplot(x='occupation', y='income', data=data)

ax.set(ylabel='Fraction of people with income > $50k')
print(data['native.country'].value_counts())

not_from_US = np.sum(data['native.country']!='United-States')

print(not_from_US, 'people are not from the United States')
data['native.country'] = (data['native.country']=='United-States')*1

#data['US_or_not']=np.where(data['native.country']=='United-States',1,0)
data.select_dtypes(exclude=[np.number]).head()
#Replace all '?'s with NaNs.

data = data.applymap(lambda x: np.nan if x=='?' else x)
data.isnull().sum(axis=0) # How many issing values are there in the dataset?
data.shape[0] - data.dropna(axis=0).shape[0]   # how many rows will be removed if I remove all the NaN's?
data = data.dropna(axis=0) ## Drop all the NaNs
#data.education.value_counts()  # I will label-encode the education column since it is an ordinal categorical variable
## This computes the fraction of people by country who earn >50k per annum

#mean_income_bycountry_df = data[['native.country','income']].groupby(['native.country']).mean().reset_index()
#edu_encode_dict = {'Preschool':0,'1st-4th':1, '5th-6th':2, '7th-8th':3, '9th':4, '10th':5,

#                  '11th':6, '12th':7, 'HS-grad':8, 'Some-college':9, 'Bachelors':10, 'Masters':11, 'Assoc-voc':12, 

#                   'Assoc-acdm':13, 'Doctorate':14, 'Prof-school':15}



#data['education'] = data['education'].apply(lambda ed_level: edu_encode_dict[ed_level])
data = pd.get_dummies(data,columns=['workclass','sex', 'marital.status',

                                    'race','relationship','occupation'],

               prefix=['workclass', 'is', 'is', 'race_is', 'relation', 'is'], drop_first=True)

### native country is ignored because that feature will be dropped later
plt.figure(figsize=(20,12))

sns.heatmap(data.corr())
data.select_dtypes(exclude=[np.number]).shape
data.groupby('income').mean()
data.shape
y = data.income

X = data.drop(['income', 'education', 'native.country', 'fnlwgt'],axis=1)
from sklearn.model_selection import train_test_split
# Split the dataset into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier as xgb

from sklearn import metrics
baseline_train = np.zeros(y_train.shape[0])

baseline_test = np.zeros(y_test.shape[0])

print('Accuracy on train data: %f%%' % (metrics.accuracy_score(y_train, baseline_train)))

print('Accuracy on test data: %f%%' %  (metrics.accuracy_score(y_test, baseline_test)))
rfmodel = RandomForestClassifier(n_estimators=300,oob_score=True,min_samples_split=5,max_depth=10,random_state=10)

rfmodel.fit(X_train,y_train)

print(rfmodel)
def show_classifier_metrics(clf, y_train=y_train,y_test=y_test, print_classification_report=True, print_confusion_matrix=True):

    print(clf)

    if print_confusion_matrix:

        print('confusion matrix of training data')

        print(metrics.confusion_matrix(y_train, clf.predict(X_train)))

        print('confusion matrix of test data')

        print(metrics.confusion_matrix(y_test, clf.predict(X_test)))

    if print_classification_report:

        print('classification report of test data')

        print(metrics.classification_report(y_test, clf.predict(X_test)))

    print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, clf.predict(X_test))*100))

    print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train, clf.predict(X_train))*100))

    print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, clf.predict(X_test))))
show_classifier_metrics(rfmodel,y_train)

print('oob score = %f'% rfmodel.oob_score_)
importance_list = rfmodel.feature_importances_

name_list = X_train.columns

importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))

plt.figure(figsize=(20,10))

plt.barh(range(len(name_list)),importance_list,align='center')

plt.yticks(range(len(name_list)),name_list)

plt.xlabel('Relative Importance in the Random Forest')

plt.ylabel('Features')

plt.title('Relative importance of Each Feature')

plt.show()
from sklearn.model_selection import cross_val_score, GridSearchCV
def grid_search(clf, parameters, X, y, n_jobs= -1, n_folds=4, score_func=None):

    if score_func:

        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func,verbose =2)

    else:

        print('Doing grid search')

        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds, verbose =1)

    gs.fit(X, y)

    print("mean test score (weighted by split size) of CV rounds: ",gs.cv_results_['mean_test_score'] )

    print ("Best parameter set", gs.best_params_, "Corresponding mean CV score",gs.best_score_)

    best = gs.best_estimator_

    return best
rfmodel2 = RandomForestClassifier(min_samples_split=5,oob_score=True, n_jobs=-1,random_state=10)

parameters = {'n_estimators': [100,200,300], 'max_depth': [10,13,15,20]}

rfmodelCV = grid_search(rfmodel2, parameters,X_train,y_train)
rfmodelCV.fit(X_train,y_train)

show_classifier_metrics(rfmodelCV,y_train)

print('oob score = %f'% rfmodelCV.oob_score_)
from xgboost.sklearn import XGBClassifier
param = {}

param['learning_rate'] = 0.1

param['verbosity'] = 1

param['colsample_bylevel'] = 0.9

param['colsample_bytree'] = 0.9

param['subsample'] = 0.9

param['reg_lambda']= 1.5

param['max_depth'] = 5

param['n_estimators'] = 400

param['seed']=10

xgb= XGBClassifier(**param)

xgb.fit(X_train, y_train, eval_metric=['error'], eval_set=[(X_train, y_train),(X_test, y_test)],early_stopping_rounds=40)
show_classifier_metrics(xgb,y_train)
importance_list = xgb.feature_importances_

name_list = X_train.columns

importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))

plt.figure(figsize=(20,10))

plt.barh(range(len(name_list)),importance_list,align='center')

plt.yticks(range(len(name_list)),name_list)

plt.xlabel('Relative Importance in XGBoost')

plt.ylabel('Features')

plt.title('Relative importance of Each Feature')

plt.show()
xgbmodel2 = XGBClassifier(seed=42)

param = {

'learning_rate': [0.1],#[0.1,0.2],

#'verbosity': [1],

'colsample_bylevel': [0.9],

'colsample_bytree': [0.9],

'subsample' : [0.9],

'n_estimators': [300],

'reg_lambda': [1.5,2,2.5],

'max_depth': [3,5,7],

 'seed': [10]   

}

xgbCV = grid_search(xgbmodel2, param,X_train,y_train)
xgbCV.fit(X_train, y_train, eval_metric=['error'], eval_set=[(X_train, y_train),(X_test, y_test)],early_stopping_rounds=40)
show_classifier_metrics(xgbCV,y_train)
#X_test.iloc[np.where(y_test != xgbCV.predict(X_test))]
from sklearn.linear_model import LogisticRegression
param = {

'C': [3,5,10], 

'verbose': [1],

    'max_iter': [100,200,500,700]

}   

logreg = LogisticRegression(random_state=10)

logreg_grid = grid_search(logreg, param, X_train,y_train, n_folds=3)
logreg_grid.fit(X_train, y_train)
show_classifier_metrics(logreg_grid)
from sklearn.naive_bayes import GaussianNB
NBmodel = GaussianNB()
NBmodel.fit(X_train, y_train)
NBmodel.predict(X_test)
show_classifier_metrics(NBmodel,y_train)
def create_stacked_dataset(clfs,modelnames, X_train=X_train,X_test=X_test):

    X_train_stack, X_test_stack = X_train, X_test

    for clf,modelname in zip(clfs,modelnames):

        temptrain = pd.DataFrame(clf.predict(X_train),index = X_train.index,columns=[modelname+'_prediction'])

        temptest  = pd.DataFrame(clf.predict(X_test),index = X_test.index,columns=[modelname+'_prediction'])

        X_train_stack = pd.concat([X_train_stack, temptrain], axis=1)

        X_test_stack = pd.concat([X_test_stack, temptest], axis=1)

    return (X_train_stack,X_test_stack)
X_train_stack,X_test_stack = create_stacked_dataset([rfmodelCV,logreg_grid,xgbCV],modelnames=['rfmodel','logreg', 'xgb'])
X_train_stack.head(2)
param = {}

param['learning_rate'] = 0.1

param['verbosity'] = 1

param['colsample_bylevel'] = 0.9

param['colsample_bytree'] = 0.9

param['subsample'] = 0.9

param['reg_lambda']= 1.5

param['max_depth'] = 5#10

param['n_estimators'] = 400

param['seed']=10

xgbstack= XGBClassifier(**param)

xgbstack.fit(X_train_stack, y_train, eval_metric=['error'], eval_set=[(X_train_stack, y_train),(X_test_stack, y_test)],early_stopping_rounds=30)

print(metrics.classification_report(y_test, xgbstack.predict(X_test_stack)))

print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, xgbstack.predict(X_test_stack))*100))

print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train, xgbstack.predict(X_train_stack))*100))
xgbstackCV = XGBClassifier(seed=10)

param_grid = {}

param_grid['learning_rate'] = [0.1]

param_grid['colsample_bylevel'] = [0.9]

param_grid['colsample_bytree'] = [0.9]

param_grid['subsample'] = [0.9]

param_grid['n_estimators'] = [300]

param_grid['reg_lambda']= [1.5]

param_grid['seed'] =[10]

param_grid['max_depth'] = [3,5,8,10]

xgbstackCV_grid = grid_search(xgbstackCV, param_grid,X_train_stack,y_train)
xgbstackCV_grid.fit(X_train_stack, y_train, eval_metric=['error'], eval_set=[(X_train_stack, y_train),(X_test_stack, y_test)],early_stopping_rounds=30)
print(metrics.classification_report(y_test, xgbstack.predict(X_test_stack)))

print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, xgbstack.predict(X_test_stack))*100))

print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train, xgbstack.predict(X_train_stack))*100))
from catboost import CatBoostClassifier
catb = CatBoostClassifier(learning_rate=0.3,iterations=400,verbose=0,random_seed=10,eval_metric='Accuracy',rsm=0.9)
catb.fit(X_train,y_train,eval_set=[(X_train,y_train), (X_test,y_test)],early_stopping_rounds=40)
show_classifier_metrics(catb)
### Catboost grid search



catbCV = CatBoostClassifier(verbose=0,random_seed=10,eval_metric='Accuracy')

param_grid = {}

param_grid['learning_rate'] = [0.1]#, 0.3]

param_grid['rsm'] = [0.9]

#param_grid['subsample'] = [0.9]

param_grid['iterations'] = [200,300]

param_grid['reg_lambda']= [3] #2

param_grid['depth'] = [8,10]#5

catbCV_grid = grid_search(catbCV, param_grid,X_train,y_train)
catbCV_grid.fit(X_train,y_train,eval_set=[(X_train,y_train), (X_test,y_test)],early_stopping_rounds=30)
show_classifier_metrics(catbCV_grid)
from imblearn.over_sampling import RandomOverSampler
np.sum(y_train)/y_train.shape[0]
ros = RandomOverSampler(random_state=1,sampling_strategy=0.8)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
catb_ros = CatBoostClassifier(learning_rate=0.1,iterations=400,reg_lambda=2,verbose=0,random_seed=10,eval_metric='Accuracy')
catb_ros.fit(X_resampled,y_resampled,eval_set=[(X_resampled,y_resampled), (X_test,y_test)],early_stopping_rounds=40)
print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, catb_ros.predict(X_test))*100))

print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_resampled, catb_ros.predict(X_resampled))*100))

print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, catb_ros.predict(X_test))))
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=10,sampling_strategy=0.7)

X_train_smt, y_train_smt = smt.fit_sample(X_train, y_train)
y_train.value_counts()
np.bincount(y_train_smt)
catb_smote = CatBoostClassifier(learning_rate=0.1,iterations=400,reg_lambda=2,verbose=0,random_seed=10,eval_metric='Accuracy')
catb_smote.fit(X_train_smt,y_train_smt,eval_set=[(X_train_smt,y_train_smt), (X_test,y_test)],early_stopping_rounds=40)
print(metrics.classification_report(y_test, catb_smote.predict(X_test)))

print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, catb_smote.predict(X_test))*100))

print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train_smt, catb_smote.predict(X_train_smt))*100))

print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, catb_ros.predict(X_test))))
param = {}

param['learning_rate'] = 0.1

param['verbosity'] = 1

param['colsample_bylevel'] = 0.9

param['colsample_bytree'] = 0.9

param['subsample'] = 0.9

param['reg_lambda']= 1.5

param['max_depth'] = 5

param['n_estimators'] = 400

param['seed']=10

xgb_smote= XGBClassifier(**param)

xgb_smote.fit(X_train_smt, y_train_smt, eval_metric=['error'], eval_set=[(X_train_smt, y_train_smt),(X_test.values, y_test.values)],early_stopping_rounds=30)
print(metrics.classification_report(y_test, xgb_smote.predict(X_test.values)))

print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, xgb_smote.predict(X_test.values))*100))

print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train_smt, xgb_smote.predict(X_train_smt))*100))

print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, xgb_smote.predict(X_test.values))))