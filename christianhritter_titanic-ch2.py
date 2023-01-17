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



from sklearn.model_selection import train_test_split, ShuffleSplit,cross_validate

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegression

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

import xgboost as xgb







import seaborn as sns

import matplotlib.pyplot as plt





# Any results you write to the current directory are saved as output.
data_train_raw = pd.read_csv("/kaggle/input/titanic-ch1/train_chI.csv")

data_test = pd.read_csv("/kaggle/input/titanic-ch1/test_chI.csv")
data_train = data_train_raw.copy(deep = True)



#however passing by reference is convenient, because we can clean both datasets at once

data_cleaner = [data_train, data_test]

data_train.info()

data_test.info()

data_train.sample(10,random_state=42)
X = data_train.drop(['Survived'],axis=1)

y = data_train['Survived']

X.shape,y.shape
X_train, X_val, y_train, y_val = train_test_split(X,y, random_state = 42)

X_train.shape,X_val.shape
# majority classifier performance:

y_train.value_counts(normalize=True),y_val.value_counts(normalize=True)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_train_pred = logreg.predict(X_train)

y_val_pred = logreg.predict(X_val)
print(classification_report(y_train,y_train_pred))

print(classification_report(y_val,y_val_pred))
coeff_df = pd.DataFrame(X_train.columns)

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#Machine Learning Algorithm (MLA) Selection and Initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]
#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit

#note: this is an alternative to train_test_split

cv_split = ShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = 0 ) # could leave out 10%


#create table to compare MLA metrics

MLA_columns = ['Name','Train Acc Mean','Parameters','Val Acc Mean', 'Val Acc 3*STD', 'Training Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)

MLA_compare
X.shape,y.shape
%%time



#create table to compare MLA predictions

MLA_predict = y.copy(deep=True) #data1[Target]



#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'Name'] = MLA_name

    MLA_compare.loc[row_index, 'Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

    cv_results = cross_validate(alg, X, y, cv  = cv_split, return_train_score=True)



    MLA_compare.loc[row_index, 'Training Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'Train Acc Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'Val Acc Mean'] = cv_results['test_score'].mean()   

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'Val Acc 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    #save MLA predictions - see section 6 for usage

    alg.fit(X, y)

    MLA_predict[MLA_name] = alg.predict(X)

    

    row_index+=1
cv_results
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

MLA_compare.sort_values(by = ['Val Acc Mean'], ascending = False, inplace = True)

MLA_compare['Variance Error'] = MLA_compare['Train Acc Mean'] - MLA_compare['Val Acc Mean']

MLA_compare[['Name','Train Acc Mean','Val Acc Mean', 'Val Acc 3*STD', 'Training Time','Variance Error']]

#MLA_predict

#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

sns.barplot(x='Val Acc Mean', y = 'Name', data = MLA_compare, color = 'm')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
parameters_xgboost = {

    'n_estimators': (250, 500, 750),# number of trees to fit

    'max_depth': (4, 6, 8,10), # Maximum tree depth for base learners. Most important feature

    'min_child_weight': (1, 5, 10), # Minimum sum of instance weight(hessian) needed in a child.

    'alpha': (0.00001, 0.000001), # L1 regularization term on weights. 

    'learning_rate': (0.1, 0.01) # learning rate

    # 'clf_reg_lambda': () # L2 regularization term on weights

    #'clf__max_iter': (10, 50, 80),

}
#gs_clf = RandomizedSearchCV(xgbmodel, parameters_xgboost, cv=cv_split, iid=False, n_jobs=1,

#                           random_state = 42, n_iter=50, verbose=3, scoring = 'accuracy') #return_training_score=True,

#                           #verbose = 0)#,n_jobs= -1)
from sklearn.model_selection import ParameterGrid

import time
#create table to compare MLA metrics

param_search_results_cols = ['Train Acc Mean','Val Acc Mean', 'Val Error STD', 'Variance Error','Training Time','Parameters']

param_search_results = pd.DataFrame(columns = param_search_results_cols)

param_search_results
%%time

row_index= 0

print('start ',len(list(ParameterGrid(parameters_xgboost))), ' run')

for params in ParameterGrid(parameters_xgboost):

    

#     xgbmodel = XGBClassifier(booster='gbtree',

#                              learning_rate=params['learning_rate'],

#                              n_estimators=params['n_estimators'],

#                              max_depth=params['max_depth'],

#                              min_child_weight = params['min_child_weight'],

#                              alpha = params['alpha'],

#                              verbosity=0, 

#                              silent=True, verbose_eval=0,random_state=42)

    data_dmat = xgb.DMatrix(data=X, label=y)

    # last eval_metric entry will be used for early stopping

    params_all={"max_depth":params['max_depth'], "min_child_weight":params['min_child_weight'], 

            "eta": params['alpha'], "eval_metric": ["error","logloss"],"objective" : "binary:logistic"}

            #"subsamples":0.9, "colsample_bytree":0.8, "objective" : "binary:logistic", "eval_metric": "logloss"}

    rounds = 180 # maximum number of boosting iterations

    t0 = time.time()

    cv_results = xgb.cv(params=params_all, dtrain=data_dmat, num_boost_round=rounds, early_stopping_rounds=50, 

                folds=cv_split, seed=23333)

    t1 = time.time()

    # values from final iteration

    final_iter = cv_results.iloc[-1]

    

    param_search_results.loc[row_index, 'Train Acc Mean'] = 1. - final_iter['train-error-mean']

    param_search_results.loc[row_index, 'Val Acc Mean'] = 1. - final_iter['test-error-mean'] 

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    param_search_results.loc[row_index, 'Val Error STD'] = final_iter['test-error-std']   #let's know the worst that can happen!

    

    param_search_results.loc[row_index, 'Parameters'] = str(params)

    param_search_results.loc[row_index, 'Training Time'] = t1-t0

    

    #param_search_results.append([final_iter['test-error-mean']])

    

    row_index +=1

    

    #if row_index>3:

    #    break



param_search_results.sort_values(by = ['Val Acc Mean'], ascending = False, inplace = True)

param_search_results['Variance Error'] = param_search_results['Train Acc Mean'] - param_search_results['Val Acc Mean']

param_search_results.head()
param_search_results.to_csv('xgb-random-grid-search-results.csv', index=False)
fig, ax1 = plt.subplots(1, 1, figsize=(18, 4))

sns.distplot(param_search_results['Train Acc Mean'].astype(float),ax=ax1,label='Train Acc Mean',kde=False)

sns.distplot(param_search_results['Val Acc Mean'].astype(float),ax=ax1,label='Val Acc Mean',kde=False)

plt.legend();plt.title("Parameter Grid Accuracy Distribution")
fig, ax1 = plt.subplots(1, 1, figsize=(18, 4))

ax1.errorbar(range(len(param_search_results)), param_search_results['Val Acc Mean'], 

                yerr=param_search_results['Val Error STD'], fmt='o')

plt.title("Parameter Grid Validation Accuracy Distribution /w STD")
param_search_results.iloc[0]['Parameters']
# Define the model

xgbmodel_final = XGBClassifier(booster='gbtree',learning_rate=0.1,n_estimators=500,max_depth=5, eta=1e-5,

                               min_child_weight = 1, verbosity=0, silent=True, verbose_eval=0,random_state=42)
%%time

%%capture cap --no-stderr

xgbmodel_final.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_val, y_val)],early_stopping_rounds=50,

          eval_metric=['error','auc','logloss'])

with open('xgboost_final_train.log', 'w') as f:

    f.write(cap.stdout)
results = xgbmodel_final.evals_result()



epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()





# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train',marker='o')

ax.plot(x_axis, results['validation_1']['error'], label='Test',marker='o')

ax.legend()

plt.ylabel('Classification Accuracy Error')

plt.title('XGBoost Classification Accuracy Error')

plt.show()



# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['auc'], label='Train',marker='o')

ax.plot(x_axis, results['validation_1']['auc'], label='Test',marker='o')

ax.legend()

plt.ylabel('AUC  Error')

plt.title('XGBoost AUC Error')

plt.show()
y_pred_test= xgbmodel_final.predict(data_test)
data_test.head()
PassengerId = pd.read_csv("/kaggle/input/titanic/test.csv").PassengerId
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_pred_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")