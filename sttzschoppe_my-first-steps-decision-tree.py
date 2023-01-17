# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import matplotlib.pyplot as plt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv("../input/train.csv")

test =  pd.read_csv("../input/test.csv")



#train.head()

all_data = pd.concat((train.loc[:,'Pclass':'Embarked'],

                      test.loc[:,'Pclass':'Embarked']))

all_data.info()

#train.head()
#filling NA's with the proper values for each column:

all_data.Age = all_data.Age.fillna(all_data.Age.median())

all_data.Fare = all_data.Fare.fillna(all_data.Fare.median())

all_data.Embarked = all_data.Embarked.fillna(all_data.Embarked.mode()[0],)



#all_data = all_data.fillna(all_data.mean())

all_data.info()
all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

#cleanup rare title names

#print(data1['Title'].value_counts())

stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (all_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

all_data['Title'] = all_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(all_data['Title'].value_counts())



all_data = all_data.drop(['Name'], axis=1)

all_data = all_data.drop(['Ticket'], axis=1)

all_data = all_data.drop(['Cabin'], axis=1)



all_data.info()
all_data = pd.get_dummies(all_data)



all_data.head()
#split dataset in cross-validation with this splitter class: 

#http://scikit-learn.org/stable/modules/generated/

#sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit

#note: this is an alternative to train_test_split

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, 

                                        train_size = .6, random_state = 0 ) 

# run model 10x with 60/30 split intentionally leaving out 10%
train_cleared = all_data[:train.shape[0]]

train_cleared.info()



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

train_cleared, train.Survived, random_state=0, test_size=0.1)



X_train.info()



X_val = all_data[train.shape[0]:]

X_val.info()
#base model

dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, train_cleared, train.Survived, cv  = cv_split, return_train_score=True)

dtree.fit(train_cleared, train.Survived)



print('BEFORE DT Parameters: ', dtree.get_params())

print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

#print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))

print('-'*10)





#tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini

              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best

              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none

              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2

              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1

              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all

              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

             }



#print(list(model_selection.ParameterGrid(param_grid)))



#choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, 

                                          scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train_cleared, train.Survived)



#print(tune_model.cv_results_.keys())

#print(tune_model.cv_results_['params'])

print('AFTER DT Parameters: ', tune_model.best_params_)

#print(tune_model.cv_results_['mean_train_score'])

print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

#print(tune_model.cv_results_['mean_test_score'])

print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
#base model

print('BEFORE DT RFE Training Shape Old: ', train_cleared.shape) 

print('BEFORE DT RFE Training Columns Old: ', train_cleared.columns.values)



print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print('-'*10)







#feature selection

dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)

dtree_rfe.fit(train_cleared, train.Survived)



#transform x&y to reduced features and fit new model

#alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

X_rfe = train_cleared.columns.values[dtree_rfe.get_support()]

rfe_results = model_selection.cross_validate(dtree, train_cleared[X_rfe], train.Survived, cv  = cv_split)



#print(dtree_rfe.grid_scores_)

print('AFTER DT RFE Training Shape New: ', train_cleared[X_rfe].shape) 

print('AFTER DT RFE Training Columns New: ', X_rfe)



print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 

print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))

print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))

print('-'*10)





#tune rfe model

rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

rfe_tune_model.fit(train_cleared[X_rfe], train.Survived)



#print(rfe_tune_model.cv_results_.keys())

#print(rfe_tune_model.cv_results_['params'])

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)

#print(rfe_tune_model.cv_results_['mean_train_score'])

print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

#print(rfe_tune_model.cv_results_['mean_test_score'])

print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
model = DecisionTreeClassifier(random_state=0, max_depth=5)



model.fit(X_train, y_train)

print("Train score: {:.3f}".format(model.score(X_train, y_train))) #0.869

print("Test score: {:.3f}".format(model.score(X_test, y_test))) #0.822

decision_tree_predicts_base = model.predict(X_val)

decision_tree_predicts_tuned_param = tune_model.predict(X_val)

decision_tree_predicts_tuned_param_rfe = rfe_tune_model.predict(X_val[X_rfe])
import graphviz 

dot_data = tree.export_graphviz(model, out_file=None, 

                         feature_names=list(train_cleared),    

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph
result = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":decision_tree_predicts_base})

result.to_csv("DecisionTree_base.csv", index = False)



result = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":decision_tree_predicts_tuned_param})

result.to_csv("DecisionTree_tuned_Param.csv", index = False)



result = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":decision_tree_predicts_tuned_param_rfe})

result.to_csv("DecisionTree_tuned_Param_RFE.csv", index = False)



result.info()

#print(check_output(["ls"]).decode("utf8"))