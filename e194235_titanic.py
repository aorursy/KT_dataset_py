# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

import re as re

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

training = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
dataset = [training, test]

print (training.info())

training.sample(5)

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
print('Train columns with null values:\n', training.isnull().sum())
print("-"*25)

print('Test columns with null values:\n', test.isnull().sum())
print("-"*25)

training.describe(include = 'all')
for data in dataset:    
    #complete missing age with median
    data['Age'].fillna(data['Age'].median(), inplace = True)

    #complete embarked with mode
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    data['Fare'].fillna(data['Fare'].median(), inplace = True)
for data in dataset:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

for data in dataset:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    
for data in dataset:
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
for data in dataset:
    data['LastName'] = data['Name'].apply(lambda x: str.split(x, ",")[0])

all_data = training.append(test)
DEFAULT_SURVIVAL_VALUE = 0.5
all_data['GroupSurvival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in all_data[['Survived', 'LastName', 'Fare', 'PassengerId'
                           ]].groupby(['LastName', 'Fare']):
    
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                all_data.loc[all_data['PassengerId'] == passID, 'GroupSurvival'] = 1
            elif (smin==0.0):
                all_data.loc[all_data['PassengerId'] == passID, 'GroupSurvival'] = 0
    

print("Number of passengers with family survival information:", 
      all_data.loc[all_data['GroupSurvival']!=0.5].shape[0])

for _, grp_df in all_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['GroupSurvival'] == 0) | (row['GroupSurvival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'GroupSurvival'] = 1
                elif (smin==0.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'GroupSurvival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(all_data[all_data['GroupSurvival']!=0.5].shape[0]))

# # Family_Survival in TRAIN_DF and TEST_DF:
training['GroupSurvival'] = all_data['GroupSurvival'][:891]
test['GroupSurvival'] = all_data['GroupSurvival'][891:]


training.sample(5)

pd.crosstab(training['Title'], training['Sex'])
for data in dataset:
    data['Title'] = data['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Vip')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
print(training['Title'].value_counts())
training[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
for data in dataset:
    data['FareBin'] = pd.qcut(data['Fare'], 5)
    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)
training.sample(5)


plt.subplot(231)
plt.hist(x = [training[training['Survived']==1]['FamilySize'], training[training['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()



plt.subplot(232)
plt.hist(x = [training[training['Survived']==1]['Sex'], training[training['Survived']==0]['Sex']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.xlabel('Sex')
plt.ylabel('# of Passengers')
plt.legend()





plt.subplot(233)
plt.hist(x = [training[training['Survived']==1]['Fare'], training[training['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()
plt.subplot(234)
plt.hist(x = [training[training['Survived']==1]['Title'], training[training['Survived']==0]['Title']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Title Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
plt.subplot(234)
plt.hist(x = [training[training['Survived']==1]['GroupSurvival'], training[training['Survived']==0]['GroupSurvival']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Title Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
plt.subplot(235)

sns.barplot(x = 'GroupSurvival', y = 'Survived', data=training)
plt.subplot(236)
plt.hist(x = [training[training['Survived']==1]['Age'], training[training['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.xlabel('Age')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(231)
plt.hist(x = [training[training['Survived']==1]['Pclass'], training[training['Survived']==0]['Pclass']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.xlabel('Class')
plt.ylabel('# of Passengers')
plt.legend()
label = LabelEncoder()
for data in dataset:    
    data['Title'] = label.fit_transform(data['Title'])
    data['Sex'] = label.fit_transform(data['Sex'])
    data['Embarked'] = label.fit_transform(data['Embarked'])
    data['FareBin'] = label.fit_transform(data['FareBin'])
    data['AgeBin'] = label.fit_transform(data['AgeBin'])

training.sample(5)
drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']
training = training.drop(drop_features, axis=1)
test = test.drop(drop_features, axis=1)
training.sample(5)

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'FareBin', 'AgeBin','GroupSurvival']
for f in features:
        print('Survival by:', f)
        print(training[[f, 'Survived']].groupby(f, as_index=False).mean())
        print('-'*25, '\n')




#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.savefig('correlations.png')
correlation_heatmap(training)
drop_features = ['IsAlone', 'Parch', 'SibSp','LastName']
training  = training.drop(drop_features, axis=1)
test = test.drop(drop_features, axis=1)
features=['Pclass', 'Sex', 'Embarked', 'FamilySize', 'FareBin', 'AgeBin']

training.sample(5)



training.info()
test.info()

features = ['Pclass','Sex' , 'Title','Embarked' ,'FamilySize', 'FareBin', 'AgeBin','GroupSurvival']
#training_x, test_x, training_y, test_y = model_selection.train_test_split(training[features], training[['Survived']],random_state=0)

#print("Training data shape:",training_x.shape)
#print("Test data shape:",test_x.shape)

#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    
    
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.Perceptron(),
    
    #Navies Bayes
   
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    
    
    #Trees    
    tree.DecisionTreeClassifier(),
    
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]


cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3,train_size=0.7, random_state = 0 )
#from sklearn.metrics import accuracy_score, log_loss
MLA_columns = ['Algorithm Name', 'Training Accuracy Mean', 'Test Accuracy Mean']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
MLA_predict = training[['Survived']]

for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'Algorithm Name'] = MLA_name
   
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, training[features], training['Survived'], cv  = cv_split)
    
    
    MLA_compare.loc[row_index, 'Training Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()   
    
    
    #alg.fit(training[features],training['Survived'])
    #res = alg.predict(training[features])
    #acc = accuracy_score(training['Survived'], res)
    #print(acc)
    #save MLA predictions - see section 6 for usage
    alg.fit(training[features], training['Survived'])
    MLA_predict[MLA_name] = alg.predict(training[features])
    
    row_index+=1
    
MLA_compare.sort_values(by = ['Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
#sns.barplot(x='Test Accuracy Mean', y = 'Algorithm Name', data = MLA_compare, color = "blue")

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
#plt.title('Algorithm Accuracy Score \n')
#plt.xlabel('Accuracy Score ')
#plt.ylabel('Algorithm')
correlation_heatmap(MLA_predict)
training.sample(5)

#all_data.sample(5)
vote_est = [
    ('xgb', XGBClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    ('ada', ensemble.AdaBoostClassifier()),
    ('lda', discriminant_analysis.LinearDiscriminantAnalysis()),
    ('lr', linear_model.LogisticRegressionCV()),
    ('svc', svm.SVC(probability=True)),
    ('dt', tree.DecisionTreeClassifier()),
    ('knn', neighbors.KNeighborsClassifier())  
]

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, training[features], training['Survived'], cv  = cv_split)
vote_hard.fit(training[features], training['Survived'])

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)


vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, training[features], training['Survived'], cv  = cv_split)
vote_soft.fit(training[features], training['Survived'])

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
    
            [{
            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
            #'learning_rate': grid_learn, #default: .3
            #'max_depth': [1,2,4,6,8,10], #default 2
            'n_estimators': grid_n_estimator, 
            'seed': grid_seed  
             }],

            [{
            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            #'loss': ['deviance', 'exponential'], #default=’deviance’ 1111
            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”   1111
            #'max_depth': grid_max_depth, #default=3
            'max_depth': [2],    
            'random_state': grid_seed
             }],

    
            [{
            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            #'n_estimators': grid_n_estimator, #default=10
            'n_estimators': [100],
            #'criterion': grid_criterion, #default=”gini”
             'criterion': ['entropy'],   
            #'max_depth': grid_max_depth, #default=None
            'max_depth': [6],
            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
             }],
    
            
            [{
            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            #'n_estimators': grid_n_estimator, #default=50
            'n_estimators': [100],
            #'learning_rate': grid_learn, #default=1
            'learning_rate': [0.25],
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R    11111
            'random_state': grid_seed
            }],
    
            #LinearDiscriminant - 
            [{}],
    
            [{
            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
            #'fit_intercept': grid_bool, #default: True
            #'penalty': ['l1','l2'],  #1111111
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
            'random_state': grid_seed
             }],
            
            
            
        
            
    
            [{
            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # 111111
            #'C': [1,2,3,4,5], #default=1.0
            #'gamma': grid_ratio, #edfault: auto
            'gamma': [0.1],
            #'decision_function_shape': ['ovo', 'ovr'], #default:ovr
            'decision_function_shape': ['ovo'],
            'probability': [True],
            'random_state': grid_seed
             }],
    
            [{
                'criterion': ['gini', 'entropy'],
                'max_depth': [2,4,6,8,10,None],
                'random_state': [0]
            }],
    
    
            [{
            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'n_neighbors': [1,2,3,4,5,6,7], #default: 5
            'weights': ['uniform', 'distance'], #default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }],

    
              
    ]

start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
    #print(param)
    
    
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
    best_search.fit(training[features], training['Survived'])
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)

grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, training[features], training['Survived'], cv  = cv_split)
grid_hard.fit(training[features], training['Survived'])

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

#Soft Vote or weighted probabilities w/Tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, training[features], training['Survived'], cv  = cv_split)
grid_soft.fit(training[features], training['Survived'])

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
print('-'*10)
MLA_columns = ['Algorithm Name', 'Training Accuracy Mean', 'Test Accuracy Mean']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
MLA_predict = training[['Survived']]

for alg in vote_est:

    #set name and parameters
    MLA_name = alg[1].__class__.__name__
    MLA_compare.loc[row_index, 'Algorithm Name'] = MLA_name
   

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg[1], training[features], training['Survived'], cv  = cv_split)
    
    
    MLA_compare.loc[row_index, 'Training Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()   
    
    
    #alg.fit(training[features],training['Survived'])
    #res = alg.predict(training[features])
    #acc = accuracy_score(training['Survived'], res)
    #print(acc)
    #save MLA predictions - see section 6 for usage
    
    
    row_index+=1
    
MLA_compare.sort_values(by = ['Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#y_pred = grid_hard.predict(test[features])
vote_est[7][1].fit(training[features], training['Survived'])
y_pred = vote_est[7][1].predict(test[features])
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("submission.csv", index = False)


