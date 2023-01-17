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

data_df = train.append(test)
data_df['Title'] = data_df['Name']

# Cleaning name and extracting Title

for name_string in data_df['Name']:

    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

    train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=True)



# Replacing rare titles with more common ones, USING ONLY TRAINING DATA

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)

train.replace({'Title': mapping}, inplace=True)

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:

    age_to_impute = train.groupby('Title')['Age'].median()[titles.index(title)]

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

    

# Substituting Age values in TRAIN and TEST:

train['Age'] = data_df['Age'][:891]

test['Age'] = data_df['Age'][891:]



#drop title column

train = train.drop(['Title'], axis=1)

# #fill NaN values in the age column with the median of that column

# train['Age'].fillna(train['Age'].mean(), inplace = True)

# #fill test with the train mean to test

# test['Age'].fillna(train['Age'].mean(), inplace = True)



#create a new column which is the combination of the sibsp and parch column

train['FamilySize'] = train ['SibSp'] + train['Parch']

test['FamilySize'] = test ['SibSp'] + test['Parch']
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])

# data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

data_df['Fare'].fillna(train['Fare'].mean(), inplace=True) #fill with training DATA ONLY





DEFAULT_SURVIVAL_VALUE = 0.5

data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

#     print(grp_df)

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0    



                

for _, grp_df in data_df.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

                    

train['Family_Survival'] = data_df['Family_Survival'][:891]

test['Family_Survival'] = data_df['Family_Survival'][891:]
#create a new column and initialize it with 1

train['IsAlone'] = 1 #initialize to yes/1 is alone

train['IsAlone'].loc[train['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

test['IsAlone'] = 1 #initialize to yes/1 is alone

test['IsAlone'].loc[test['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1  
print(train.describe()) #used to find the bin range to split
# data_df['FareBin'] = pd.qcut(train['Fare'], 5) #split based on training data only

# label = LabelEncoder()

# data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])



#alternatively, you can split them yourselves based on the bins you prefer, and you can do the same for the age too

    #Mapping Fare

data_df.loc[ data_df['Fare'] <= 7.91, 'FareBin_Code'] 						        = 0

data_df.loc[(data_df['Fare'] > 7.91) & (data_df['Fare'] <= 14.454), 'FareBin_Code'] = 1

data_df.loc[(data_df['Fare'] > 14.454) & (data_df['Fare'] <= 31), 'FareBin_Code']   = 2

data_df.loc[ data_df['Fare'] > 31, 'FareBin_Code'] 							        = 3



train['FareBin_Code'] = data_df['FareBin_Code'][:891]

test['FareBin_Code'] = data_df['FareBin_Code'][891:]
# print(pd.qcut(train['Fare'], 5)) #if you want to split into 5, but dont forget to maximize the range so for the first and last value

#e.g. x< 7.854 instead of 0<x<7.854 and x> 39.688 instead of 39.688 > 512.329
print(pd.qcut(train['Age'], 4))
# data_df['AgeBin'] = pd.qcut(train['Age'], 4) #split using training data only

# label = LabelEncoder()

# data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])



# Mapping Age

data_df.loc[ data_df['Age'] <= 21.5, 'AgeBin_Code'] 					       = 0

data_df.loc[(data_df['Age'] > 21.5) & (data_df['Age'] <= 30), 'AgeBin_Code'] = 1

data_df.loc[(data_df['Age'] > 30) & (data_df['Age'] <= 35), 'AgeBin_Code'] = 2

data_df.loc[(data_df['Age'] > 35), 'AgeBin_Code'] = 3



train['AgeBin_Code'] = data_df['AgeBin_Code'][:891]

test['AgeBin_Code'] = data_df['AgeBin_Code'][891:]



train['Sex'].replace(['male','female'],[0,1],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)
print(train.head())
g = sns.heatmap(train[["AgeBin_Code","Pclass","Sex","IsAlone","FamilySize","Family_Survival","FareBin_Code"]].corr(),cmap="BrBG",annot=True)
#train columns to drop final

drop_column = ['PassengerId','Cabin', 'Ticket','Name', 'Embarked', 'Age','Fare', 'SibSp','Parch']

train.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)



#define y variable aka target/outcome

Target = ['Survived']
print(train)
from sklearn.preprocessing import StandardScaler



# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data

std_scaler = StandardScaler()

x_train = std_scaler.fit_transform(x_train)

x_test = std_scaler.transform(x_test)
# #test splitting

# from sklearn.model_selection import train_test_split



# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/9, random_state=42)
print(x_train.shape)
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn import model_selection



import time

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
import time

from sklearn.model_selection import cross_val_score



cv_results = []

for classifier in vote_est :

    print(classifier)

    cv_results.append(cross_val_score(classifier[1], X = x_train, y = y_train, scoring = "accuracy", cv = 10, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["AdaBoost","Bagging","ExtraTrees",

"GradientBoosting","RandomForest","GaussianProcess","LogisticRegression","BernoulliNB","GaussianNB","kNeighbors","SVC","XGBoost"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
# #Hard Vote or majority rules

# vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

# vote_hard_cv = model_selection.cross_validate(vote_hard, x_train, y_train, cv  = 10, return_train_score = True)

# vote_hard.fit(x_train, y_train)

# hard_predict = vote_hard.predict(x_test)



# print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 

# print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

# print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

# print('-'*10)





# #Soft Vote or weighted probabilities

# vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

# vote_soft_cv = model_selection.cross_validate(vote_soft, x_train, y_train, cv  = 10 , return_train_score = True)

# vote_soft.fit(x_train, y_train)

# soft_predict = vote_soft.predict(x_test)





# print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 

# print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

# print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))

# print('-'*10)
# import matplotlib.pyplot as plt

#some dont have feature_importances_, thats why wont work for all



# nrows = 3

# ncols = 2

# fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))



# nclassifier = 0

# for row in range(nrows):

#     for col in range(ncols):

#         name = vote_est[nclassifier][0]

#         classifier = vote_est[nclassifier][1]

#         vote_est[nclassifier][1].fit(x_train, y_train)

#         indices = np.argsort(classifier.feature_importances_)[::-1][:40]

#         g = sns.barplot(y=train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])

#         g.set_xlabel("Relative importance",fontsize=12)

#         g.set_ylabel("Features",fontsize=12)

#         g.tick_params(labelsize=9)

#         g.set_title(name + " feature importance")

#         nclassifier += 1
# #WARNING: Running is very computational intensive and time expensive.

# #Code is written for experimental/developmental purposes and not production ready!





# #Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# grid_learn_ada = [0.2,0.22,0.24, .25, 0.26, 0.28, 0.3]

# grid_n_estimator_ada = [250, 275, 300, 325, 350]





# grid_n_estimator = [10, 50, 100, 300]

# grid_ratio = [.1, .25, .5, .75, 1.0]

# grid_learn = [.01, .03, .05, .1, .25]



# grid_max_depth = [2, 4, 6, 8, 10, None]

# grid_min_samples = [5, 10, .03, .05, .10]

# grid_criterion = ['gini', 'entropy']

# grid_bool = [True, False]

# grid_seed = [0]





# grid_param = [

#             [{

#             #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

#             'n_estimators': grid_n_estimator_ada, #default=50

#             'learning_rate': grid_learn_ada, #default=1

#             #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R

#             'random_state': grid_seed

#             }],

       

    

#             [{

#             #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

#             'n_estimators': grid_n_estimator, #default=10

#             'max_samples': grid_ratio, #default=1.0

#             'random_state': grid_seed

#              }],



    

#             [{

#             #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier

#             'n_estimators': grid_n_estimator, #default=10

#             'criterion': grid_criterion, #default=”gini”

#             'max_depth': grid_max_depth, #default=None

#             'random_state': grid_seed

#              }],





#             [{

#             #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

#             #'loss': ['deviance', 'exponential'], #default=’deviance’

#             'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

#             'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

#             #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”

#             'max_depth': grid_max_depth, #default=3   

#             'random_state': grid_seed

#              }],



    

#             [{

#             #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

#             'n_estimators': grid_n_estimator, #default=10

#             'criterion': grid_criterion, #default=”gini”

#             'max_depth': grid_max_depth, #default=None

#             'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.

#             'random_state': grid_seed

#              }],

    

#             [{    

#             #GaussianProcessClassifier

#             'max_iter_predict': grid_n_estimator, #default: 100

#             'random_state': grid_seed

#             }],

        

    

#             [{

#             #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

#             'fit_intercept': grid_bool, #default: True

#             #'penalty': ['l1','l2'],

#             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs

#             'random_state': grid_seed

#              }],

            

    

#             [{

#             #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

#             'alpha': grid_ratio, #default: 1.0

#              }],

    

    

#             #GaussianNB - 

#             [{}],

    

#             [{

#             #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

#             'n_neighbors': [1,2,3,4,5,6,7], #default: 5

#             'weights': ['uniform', 'distance'], #default = ‘uniform’

#             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

#             }],

            

    

#             [{

#             #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

#             #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r

#             #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

#             'C': [1,2,3,4,5], #default=1.0

#             'gamma': grid_ratio, #edfault: auto

#             'decision_function_shape': ['ovo', 'ovr'], #default:ovr

#             'probability': [True],

#             'random_state': grid_seed

#              }],



    

#             [{

#             #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

#             'learning_rate': grid_learn, #default: .3

#             'max_depth': [1,2,4,6,8,10], #default 2

#             'n_estimators': grid_n_estimator, 

#             'seed': grid_seed  

#              }]   

#         ]







# start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

# for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip



#     #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm

#     #print(param)

    

    

#     start = time.perf_counter()        

#     best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = 10, scoring = 'roc_auc')

#     best_search.fit(x_train, y_train)

#     run = time.perf_counter() - start



#     best_param = best_search.best_params_

#     print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))

#     clf[1].set_params(**best_param) 





# run_total = time.perf_counter() - start_total

# print('Total optimization time was {:.2f} minutes.'.format(run_total/60))



# print('-'*10)
grid_n_estimator = [10, 50, 100, 300]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25]

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]



inner_cv = model_selection.KFold(n_splits=9, shuffle=True, random_state=0)

outer_cv = model_selection.KFold(n_splits=9, shuffle=True, random_state=0)



xgboost_grid = [{

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': [0.01,0.025,0.05,0.075,0.10], #default: .3

            'max_depth': [3,4,5], #default 2

            'n_estimators': [300,350,400,450,500], 

            'seed': grid_seed  

             }]

xgb =  XGBClassifier()

best_search = model_selection.GridSearchCV(estimator =xgb, param_grid = xgboost_grid, cv = inner_cv, scoring = 'accuracy')

best_search.fit(x_train, y_train)

#best_param = best_search.best_params_ 

print(best_search.best_estimator_)

print(best_search.best_score_)

best_search.best_estimator_.fit(x_train,y_train)



# y_pred_val = best_search.best_estimator_.predict(x_test)

y_pred= best_search.best_estimator_.predict(x_test)
# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.model_selection import GridSearchCV



# n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]

# algorithm = ['auto']

# weights = ['uniform', 'distance']

# leaf_size = list(range(1,50,5))

# hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

#                'n_neighbors': n_neighbors}

# best_search=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 

#                 cv=9, scoring = "f1_macro")

# best_search.fit(x_train, y_train)

# print(best_search.best_score_)

# print(best_search.best_estimator_)

# print(best_search.best_params_)

# best_search.best_estimator_.fit(x_train, y_train)

# y_pred = best_search.best_estimator_.predict(x_test)



# # knn = KNeighborsClassifier()

# # knn.fit(x_train,y_train)



# # y_pred = knn.predict(x_test)



# best_search_cv = model_selection.cross_validate(best_search.best_estimator_, x_train, y_train, cv  = 9, return_train_score = True, return_estimator=True,scoring='f1_macro')

best_search_cv = model_selection.cross_validate(xgb, x_train, y_train, cv  = outer_cv, return_train_score = True, return_estimator=True,scoring='f1_macro')



# from sklearn.metrics import accuracy_score

# print("ValidationScore",accuracy_score(y_val, y_pred_val))



print(best_search.best_params_)

print("Training w/bin score mean: {:.2f}". format(best_search_cv['train_score'].mean()*100)) 

print("Test w/bin score mean: {:.2f}". format(best_search_cv['test_score'].mean()*100))

print("Test w/bin score 3*std: +/- {:.2f}". format(best_search_cv['test_score'].std()*100*3))

# print("Estimayot: +/- {:.2f}". format(best_search_cv['estimator']))

print('-'*10)
from sklearn.model_selection import learning_curve



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



g = plot_learning_curve(best_search.best_estimator_,"KNN learning curves",x_train,y_train,cv=outer_cv)
#

# coeff_df = pd.DataFrame(vote_hard.columns.delete(0))

# coeff_df.columns = ['Feature']

# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



# coeff_df.sort_values(by='Correlation', ascending=False)
# #It's also possible to automatically select the optimal number of features and visualize this. 

# #This is uncommented and can be tried in the competition part of the tutorial.

# rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )

# rfecv.fit( train_X , train_y )
# # Generate Submission File 

# HardVoteSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

#                             'Survived': hard_predict })

# HardVoteSubmission.to_csv("HardVoteSubmission.csv", index=False)



# SoftVoteSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

#                             'Survived': soft_predict })

# SoftVoteSubmission.to_csv("SoftVoteSubmission.csv", index=False)

                         

titanicSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': y_pred })

print(titanicSubmission)

titanicSubmission.to_csv("titanicSubmission.csv", index=False)