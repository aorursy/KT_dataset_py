#import libraries

import pandas as pd

import numpy as np





# from sklearn.feature_selection import SelectPercentile

# from sklearn.feature_selection import chi2 , f_classif 

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.model_selection import GridSearchCV

# read and prepare  data

cric_bat = pd.read_csv('../input/cricket/output.csv')

print(cric_bat)

print(cric_bat.shape)

#number of rows and features

print('number of rows =',cric_bat.shape[0])



print('number of features =',cric_bat.shape[1])

Data_head = cric_bat.head(10)

print(Data_head)

Data_tail = cric_bat.tail(10)

print(Data_tail)



print(cric_bat.describe())

print(cric_bat.info())









#dropping columns

cric_bat.drop(['Runs' , 'Unnamed: 0'], axis=1, inplace=True)
#X Data

X = cric_bat.drop(['output'], axis=1, inplace=False)

X= pd.get_dummies(X)

#print('X Data is \n' , X.head())

#print('X shape is ' , X.shape)



#y Data

y = cric_bat['output']

#print('y Data is \n' , y.head())

#print('y shape is ' , y.shape)
#----------------------------------------------------

# Cleaning data



'''

impute.SimpleImputer(missing_values=nan, strategy='mean’, fill_value=None, verbose=0, copy=True)

'''

from sklearn.impute import SimpleImputer





ImputedModule = SimpleImputer(missing_values = 0, strategy ='mean')

ImputedX = ImputedModule.fit(X)

X = ImputedX.transform(X)





#X Data

#print('X Data is \n' , X[:10])



#y Data

#print('y Data is \n' , y[:10])

#splittig the dataset to traning set and test set

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 44,shuffle =True)
#----------------------------------------------------

#Standard Scaler for Data

# from sklearn.preprocessing import StandardScaler



# scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

# X = scaler.fit_transform(X)



#showing data

#print('X \n' , X[:10])

#print('y \n' , y[:10])
#Applying RandomForestClassifier Model 



'''

ensemble.RandomForestClassifier(n_estimators='warn’, criterion=’gini’, max_depth=None,

                                min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,

                                max_features='auto’,max_leaf_nodes=None,min_impurity_decrease=0.0,

                                min_impurity_split=None, bootstrap=True,oob_score=False, n_jobs=None,

                                random_state=None, verbose=0,warm_start=False, class_weight=None)

'''



RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=300,max_depth=3,random_state=None) #criterion can be also : entropy 

RandomForestClassifierModel.fit(X_train, y_train)



#Calculating Details

print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))

print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)

print('----------------------------------------------------')



#Calculating Prediction

y_pred = RandomForestClassifierModel.predict(X_test)

y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)

print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for RandomForestClassifierModel is : ' , y_pred_prob[:10])



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)



PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Precision Score is : ', PrecisionScore)





#----------------------------------------------------

#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2  

# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)



RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Recall Score is : ', RecallScore)



#----------------------------------------------------

# #Grid search for Random forest classifiers 

# SelectedModel = RandomForestClassifier()

# SelectedParameters = { 

#             "n_estimators"      : [100,200,300],

#             "max_depth"      : [1,2,3],

#             "min_samples_split" : [2,4,8],

#             "bootstrap": [True, False],

#             }



# # #=======================================================================

# GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)

# GridSearchModel.fit(X_train, y_train)

# sorted(GridSearchModel.cv_results_.keys())

# GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]



# # Showing Results

# print('All Results are :\n', GridSearchResults )

# print('Best Score is :', GridSearchModel.best_score_)

# print('Best Parameters are :', GridSearchModel.best_params_)

# print('Best Estimator is :', GridSearchModel.best_estimator_)
#Applying SVC Model 



'''

sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True,

                probability=False, tol=0.001, cache_size=200, class_weight=None,verbose=False,

                max_iter=-1, decision_function_shape='ovr’, random_state=None)

'''



SVCModel = SVC(kernel= 'poly',# it can be also linear,poly,sigmoid,precomputed

               max_iter=-1,C=1.0,gamma='auto')

SVCModel.fit(X_train, y_train)



#Calculating Details

print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))

print('----------------------------------------------------')



#Calculating Prediction

y_pred = SVCModel.predict(X_test)

print('Predicted Value for SVCModel is : ' , y_pred[:10])

#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)



PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Precision Score is : ', PrecisionScore)



#----------------------------------------------------

#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2  

# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)



RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Recall Score is : ', RecallScore)



#----------------------------------------------------



#----------------------------------------------------

#Applying Grid Searching :  

'''

model_selection.GridSearchCV(estimator, param_grid, scoring=None,fit_params=None, n_jobs=None, iid=’warn’,

                             refit=True, cv=’warn’, verbose=0,pre_dispatch=‘2*n_jobs’, error_score=

                             ’raisedeprecating’,return_train_score=’warn’)



'''



#=======================================================================



#=======================================================================

SelectedModel = SVC(gamma='auto')

SelectedParameters = {'kernel':('poly', 'rbf'), 'C':[1,2,3,4,5]}

GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]





# Showing Results

print('All Results are :\n', GridSearchResults )

print('Best Score is :', GridSearchModel.best_score_)

print('Best Parameters are :', GridSearchModel.best_params_)

print('Best Estimator is :', GridSearchModel.best_estimator_)

#Import Libraries

from sklearn.tree import DecisionTreeClassifier

#----------------------------------------------------



#----------------------------------------------------

#Applying DecisionTreeClassifier Model 



'''

sklearn.tree.DecisionTreeClassifier(criterion='gini’, splitter=’best’, max_depth=None,min_samples_split=2,

                                    min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,

                                    random_state=None, max_leaf_nodes=None,min_impurity_decrease=0.0,

                                    min_impurity_split=None, class_weight=None,presort=False)

'''



DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=None) #criterion can be entropy

DecisionTreeClassifierModel.fit(X_train, y_train)



#Calculating Details

print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))

print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))

print('DecisionTreeClassifierModel Classes are : ' , DecisionTreeClassifierModel.classes_)

print('DecisionTreeClassifierModel feature importances are : ' , DecisionTreeClassifierModel.feature_importances_)

print('----------------------------------------------------')



#Calculating Prediction

y_pred = DecisionTreeClassifierModel.predict(X_test)

y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)

print('Predicted Value for DecisionTreeClassifierModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for DecisionTreeClassifierModel is : ' , y_pred_prob[:10])

# # #Grid search for Decision Tree classifiers 

# SelectedModel = DecisionTreeClassifier()

# SelectedParameters = { 

#             "criterion"      : ['gini','entropy'],

#             "max_depth"      : [1,2,3],

#             "min_samples_split" : [2,4,8],

            

#             }



# # #=======================================================================

# GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)

# GridSearchModel.fit(X_train, y_train)

# sorted(GridSearchModel.cv_results_.keys())

# GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]



# # Showing Results

# print('All Results are :\n', GridSearchResults )

# print('Best Score is :', GridSearchModel.best_score_)

# print('Best Parameters are :', GridSearchModel.best_params_)

# print('Best Estimator is :', GridSearchModel.best_estimator_)





PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Precision Score is : ', PrecisionScore)



#----------------------------------------------------

#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2  

# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)



RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Recall Score is : ', RecallScore)