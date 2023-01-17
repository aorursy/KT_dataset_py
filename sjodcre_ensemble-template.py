import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold

import xgboost as xgb



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls





# Load dataset.

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



PassengerId = test['PassengerId']



#fill NaN values in the age column with the median of that column

train['Age'].fillna(train['Age'].mean(), inplace = True)

#fill test with the train mean to test

test['Age'].fillna(train['Age'].mean(), inplace = True)



#fill NaN values in the embarked column with the mode of that column

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)

#fill test NaN values in the embarked column with the mode from the train set

test['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)



#fill NaN values in the fare column with the median of that column

train['Fare'].fillna(train['Fare'].median(), inplace = True)

test['Fare'].fillna(train['Fare'].median(), inplace = True)



#delete the cabin feature/column and others 

drop_column = ['PassengerId','Cabin', 'Ticket']

train.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)



#create a new column which is the combination of the sibsp and parch column

train['FamilySize'] = train ['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test ['SibSp'] + test['Parch'] + 1



#create a new column and initialize it with 1

train['IsAlone'] = 1 #initialize to yes/1 is alone

train['IsAlone'].loc[train['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

test['IsAlone'] = 1 #initialize to yes/1 is alone

test['IsAlone'].loc[test['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



#quick and dirty code split title from the name column

train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test['Title'] = test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



#Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

#Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

train['FareBin'] = pd.qcut(train['Fare'], 4)

test['FareBin'] = pd.qcut(train['Fare'], 4)



#alternatively, you can split them yourselves based on the bins you prefer, and you can do the same for the age too

#     #Mapping Fare

#     dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

#     dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

#     dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

#     dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

#     # Mapping Age

#     dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

#     dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;



#Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

train['AgeBin'] = pd.cut(train['Age'].astype(int), 5)

test['AgeBin'] = pd.cut(train['Age'].astype(int), 5)



#so create stat_min and any titles less than 10 will be put into Misc category

stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (train['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

title_names_test = (test['Title'].value_counts() < stat_min)



#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

test['Title'] = test['Title'].apply(lambda x: 'Misc' if title_names_test.loc[x] == True else x)



#convertion from categorical data to dummy variables

label = LabelEncoder()  

train['Sex_Code'] = label.fit_transform(train['Sex'])

train['Embarked_Code'] = label.fit_transform(train['Embarked'])

train['Title_Code'] = label.fit_transform(train['Title'])

train['AgeBin_Code'] = label.fit_transform(train['AgeBin'])

train['FareBin_Code'] = label.fit_transform(train['FareBin'])



test['Sex_Code'] = label.fit_transform(test['Sex'])

test['Embarked_Code'] = label.fit_transform(test['Embarked'])

test['Title_Code'] = label.fit_transform(test['Title'])

test['AgeBin_Code'] = label.fit_transform(test['AgeBin'])

test['FareBin_Code'] = label.fit_transform(test['FareBin'])



#train columns to drop final

drop_column = ['Sex','Name', 'Embarked', 'Title','Age','Fare', 'FareBin','AgeBin']

train.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)



#define y variable aka target/outcome

Target = ['Survived']
# #you need to run one by one, but just for compilation sake I will put them together.

# train.tail()

# test.tail()

# train.isnull().sum()

# test.isnull().sum()
# # Some useful parameters which will come in handy later on

# ntrain = train.shape[0]

# ntest = test.shape[0]

# SEED = 0 # for reproducibility

# NFOLDS = 5 # set folds for out-of-fold prediction

# kf = KFold(n_splits= NFOLDS, random_state=SEED)



# # Class to extend the Sklearn classifier

# class SklearnHelper(object):

#     def __init__(self, clf, seed=0, params=None):

#         params['random_state'] = seed

#         self.clf = clf(**params)



#     def train(self, x_train, y_train):

#         self.clf.fit(x_train, y_train)



#     def predict(self, x):

#         return self.clf.predict(x)

    

#     def fit(self,x,y):

#         return self.clf.fit(x,y)

    

#     def feature_importances(self,x,y):

#        return self.clf.fit(x,y).feature_importances_
# def get_oof(clf, x_train, y_train, x_test):

#     oof_train = np.zeros((ntrain,))

#     oof_test = np.zeros((ntest,))

#     oof_test_skf = np.empty((NFOLDS, ntest))



#     for i, (train_index, test_index) in enumerate(kf.split(x_train)):

#         x_tr = x_train[train_index]

#         y_tr = y_train[train_index]

#         x_te = x_train[test_index]



#         clf.train(x_tr, y_tr)



#         oof_train[test_index] = clf.predict(x_te)

#         oof_test_skf[i, :] = clf.predict(x_test)



#     oof_test[:] = oof_test_skf.mean(axis=0)

#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# # Put in our parameters for said classifiers

# # Random Forest parameters

# rf_params = {

#     'n_jobs': -1,

#     'n_estimators': 500,

#      'warm_start': True, 

#      #'max_features': 0.2,

#     'max_depth': 6,

#     'min_samples_leaf': 2,

#     'max_features' : 'sqrt',

#     'verbose': 0

# }



# # Extra Trees Parameters

# et_params = {

#     'n_jobs': -1,

#     'n_estimators':500,

#     #'max_features': 0.5,

#     'max_depth': 8,

#     'min_samples_leaf': 2,

#     'verbose': 0

# }



# # AdaBoost parameters

# ada_params = {

#     'n_estimators': 500,

#     'learning_rate' : 0.75

# }



# # Gradient Boosting parameters

# gb_params = {

#     'n_estimators': 500,

#      #'max_features': 0.2,

#     'max_depth': 5,

#     'min_samples_leaf': 2,

#     'verbose': 0

# }



# # Support Vector Classifier parameters 

# svc_params = {

#     'kernel' : 'linear',

#     'C' : 0.025

#     }
# # Create 5 objects that represent our 4 models

# rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

# et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

# ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

# gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

# svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
# # Create our OOF train and test predictions. These base results will be used as new features

# et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

# rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

# ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

# gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
# rf_feature = rf.feature_importances(x_train,y_train)

# et_feature = et.feature_importances(x_train, y_train)

# ada_feature = ada.feature_importances(x_train, y_train)

# gb_feature = gb.feature_importances(x_train,y_train)
# cols = train.columns.values

# # Create a dataframe with features

# feature_dataframe = pd.DataFrame( {'features': cols,

#      'Random Forest feature importances': rf_feature,

#      'Extra Trees  feature importances': et_feature,

#       'AdaBoost feature importances': ada_feature,

#     'Gradient Boost feature importances': gb_feature

#     })
# # Scatter plot 

# trace = go.Scatter(

#     y = feature_dataframe['Random Forest feature importances'].values,

#     x = feature_dataframe['features'].values,

#     mode='markers',

#     marker=dict(

#         sizemode = 'diameter',

#         sizeref = 1,

#         size = 25,

# #       size= feature_dataframe['AdaBoost feature importances'].values,

#         #color = np.random.randn(500), #set color equal to a variable

#         color = feature_dataframe['Random Forest feature importances'].values,

#         colorscale='Portland',

#         showscale=True

#     ),

#     text = feature_dataframe['features'].values

# )

# data = [trace]



# layout= go.Layout(

#     autosize= True,

#     title= 'Random Forest Feature Importance',

#     hovermode= 'closest',

# #     xaxis= dict(

# #         title= 'Pop',

# #         ticklen= 5,

# #         zeroline= False,

# #         gridwidth= 2,

# #     ),

#     yaxis=dict(

#         title= 'Feature Importance',

#         ticklen= 5,

#         gridwidth= 2

#     ),

#     showlegend= False

# )

# fig = go.Figure(data=data, layout=layout)

# py.iplot(fig,filename='scatter2010')



# # Scatter plot 

# trace = go.Scatter(

#     y = feature_dataframe['Extra Trees  feature importances'].values,

#     x = feature_dataframe['features'].values,

#     mode='markers',

#     marker=dict(

#         sizemode = 'diameter',

#         sizeref = 1,

#         size = 25,

# #       size= feature_dataframe['AdaBoost feature importances'].values,

#         #color = np.random.randn(500), #set color equal to a variable

#         color = feature_dataframe['Extra Trees  feature importances'].values,

#         colorscale='Portland',

#         showscale=True

#     ),

#     text = feature_dataframe['features'].values

# )

# data = [trace]



# layout= go.Layout(

#     autosize= True,

#     title= 'Extra Trees Feature Importance',

#     hovermode= 'closest',

# #     xaxis= dict(

# #         title= 'Pop',

# #         ticklen= 5,

# #         zeroline= False,

# #         gridwidth= 2,

# #     ),

#     yaxis=dict(

#         title= 'Feature Importance',

#         ticklen= 5,

#         gridwidth= 2

#     ),

#     showlegend= False

# )

# fig = go.Figure(data=data, layout=layout)

# py.iplot(fig,filename='scatter2010')



# # Scatter plot 

# trace = go.Scatter(

#     y = feature_dataframe['AdaBoost feature importances'].values,

#     x = feature_dataframe['features'].values,

#     mode='markers',

#     marker=dict(

#         sizemode = 'diameter',

#         sizeref = 1,

#         size = 25,

# #       size= feature_dataframe['AdaBoost feature importances'].values,

#         #color = np.random.randn(500), #set color equal to a variable

#         color = feature_dataframe['AdaBoost feature importances'].values,

#         colorscale='Portland',

#         showscale=True

#     ),

#     text = feature_dataframe['features'].values

# )

# data = [trace]



# layout= go.Layout(

#     autosize= True,

#     title= 'AdaBoost Feature Importance',

#     hovermode= 'closest',

# #     xaxis= dict(

# #         title= 'Pop',

# #         ticklen= 5,

# #         zeroline= False,

# #         gridwidth= 2,

# #     ),

#     yaxis=dict(

#         title= 'Feature Importance',

#         ticklen= 5,

#         gridwidth= 2

#     ),

#     showlegend= False

# )

# fig = go.Figure(data=data, layout=layout)

# py.iplot(fig,filename='scatter2010')



# # Scatter plot 

# trace = go.Scatter(

#     y = feature_dataframe['Gradient Boost feature importances'].values,

#     x = feature_dataframe['features'].values,

#     mode='markers',

#     marker=dict(

#         sizemode = 'diameter',

#         sizeref = 1,

#         size = 25,

# #       size= feature_dataframe['AdaBoost feature importances'].values,

#         #color = np.random.randn(500), #set color equal to a variable

#         color = feature_dataframe['Gradient Boost feature importances'].values,

#         colorscale='Portland',

#         showscale=True

#     ),

#     text = feature_dataframe['features'].values

# )

# data = [trace]



# layout= go.Layout(

#     autosize= True,

#     title= 'Gradient Boosting Feature Importance',

#     hovermode= 'closest',

# #     xaxis= dict(

# #         title= 'Pop',

# #         ticklen= 5,

# #         zeroline= False,

# #         gridwidth= 2,

# #     ),

#     yaxis=dict(

#         title= 'Feature Importance',

#         ticklen= 5,

#         gridwidth= 2

#     ),

#     showlegend= False

# )

# fig = go.Figure(data=data, layout=layout)

# py.iplot(fig,filename='scatter2010')
# # Create the new column containing the average of values



# feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

# feature_dataframe.head(3)



# y = feature_dataframe['mean'].values

# x = feature_dataframe['features'].values

# data = [go.Bar(

#             x= x,

#              y= y,

#             width = 0.5,

#             marker=dict(

#                color = feature_dataframe['mean'].values,

#             colorscale='Portland',

#             showscale=True,

#             reversescale = False

#             ),

#             opacity=0.6

#         )]



# layout= go.Layout(

#     autosize= True,

#     title= 'Barplots of Mean Feature Importance',

#     hovermode= 'closest',

# #     xaxis= dict(

# #         title= 'Pop',

# #         ticklen= 5,

# #         zeroline= False,

# #         gridwidth= 2,

# #     ),

#     yaxis=dict(

#         title= 'Feature Importance',

#         ticklen= 5,

#         gridwidth= 2

#     ),

#     showlegend= False

# )

# fig = go.Figure(data=data, layout=layout)

# py.iplot(fig, filename='bar-direct-labels')
# base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

#      'ExtraTrees': et_oof_train.ravel(),

#      'AdaBoost': ada_oof_train.ravel(),

#       'GradientBoost': gb_oof_train.ravel()

#     })

# base_predictions_train.head()
# data = [

#     go.Heatmap(

#         z= base_predictions_train.astype(float).corr().values ,

#         x=base_predictions_train.columns.values,

#         y= base_predictions_train.columns.values,

#           colorscale='Viridis',

#             showscale=True,

#             reversescale = True

#     )

# ]

# py.iplot(data, filename='labelled-heatmap')
# x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

# x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
# gbm = xgb.XGBClassifier(

#     #learning_rate = 0.02,

#  n_estimators= 2000,

#  max_depth= 4,

#  min_child_weight= 2,

#  #gamma=1,

#  gamma=0.9,                        

#  subsample=0.8,

#  colsample_bytree=0.8,

#  objective= 'binary:logistic',

#  nthread= -1,

#  scale_pos_weight=1).fit(x_train, y_train)

# predictions = gbm.predict(x_test)
# # Generate Submission File 

# StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

#                             'Survived': predictions })

# StackingSubmission.to_csv("StackingSubmission.csv", index=False)
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn import model_selection



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



vote_est = [

    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html

    ('ada', ensemble.AdaBoostClassifier()),

    ('bc', ensemble.BaggingClassifier()),

    ('etc',ensemble.ExtraTreesClassifier()),

    ('gbc', ensemble.GradientBoostingClassifier()),

    ('rfc', ensemble.RandomForestClassifier()),



    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc

    ('gpc', gaussian_process.GaussianProcessClassifier()),

    

    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    ('lr', linear_model.LogisticRegressionCV()),

    

    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html

    ('bnb', naive_bayes.BernoulliNB()),

    ('gnb', naive_bayes.GaussianNB()),

    

    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html

    ('knn', neighbors.KNeighborsClassifier()),

    

    #SVM: http://scikit-learn.org/stable/modules/svm.html

    ('svc', svm.SVC(probability=True)),

    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

   ('xgb', XGBClassifier())



]
#Hard Vote or majority rules

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

vote_hard_cv = model_selection.cross_validate(vote_hard, x_train, y_train, cv  = 10, return_train_score = True)

vote_hard.fit(x_train, y_train)

hard_predict = vote_hard.predict(x_test)



print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 

print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

print('-'*10)





#Soft Vote or weighted probabilities

vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

vote_soft_cv = model_selection.cross_validate(vote_soft, x_train, y_train, cv  = 10 , return_train_score = True)

vote_soft.fit(x_train, y_train)

soft_predict = vote_soft.predict(x_test)





print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 

print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))

print('-'*10)



# Generate Submission File 

HardVoteSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': hard_predict })

HardVoteSubmission.to_csv("HardVoteSubmission.csv", index=False)



SoftVoteSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': soft_predict })

SoftVoteSubmission.to_csv("SoftVoteSubmission.csv", index=False)