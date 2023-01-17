# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import 

import numpy as np

import pandas as pd

import hyperopt

from catboost import Pool, CatBoostClassifier, cv

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
np.random.seed(123)
#get the train and test data

#train_df = pd.read_csv('../input/titanic/train.csv')

#test_df = pd.read_csv('../input/titanic/test.csv')

train_df = pd.read_pickle('../input/titanic-top-3-full-eda-model/train_x.pkl')

train_y = pd.read_pickle('../input/titanic-top-3-full-eda-model/train_y.pkl')

test_df = pd.read_pickle('../input/titanic-top-3-full-eda-model/test_x.pkl')

test_id = pd.read_pickle('../input/titanic-top-3-full-eda-model/test_id.pkl')

answer = pd.read_pickle('../input/titanic-top-3-full-eda-model/test_y.pkl')
# Show the train data

train_df.info()
# Convert Connected_Survival to int

train_df['Connected_Survival'] = train_df['Connected_Survival'].map({0:0,0.5:1,1:2})    

test_df['Connected_Survival'] = test_df['Connected_Survival'].map({0:0,0.5:1,1:2})    
# Show how many the null value for each column

train_df.isnull().sum()
"""# Fill nan with mean of the column for age

train_df['Age'].fillna(train_df['Age'].mean(),inplace=True)

test_df['Age'].fillna(test_df['Age'].mean(),inplace=True)

# for the train data, the age, are and embarked has null value, so just make it -999 for it, and Catboost will distinguish it

train_df.fillna(-999,inplace=True)

test_df.fillna(-999,inplace=True)"""
"""# Convert Age to int

train_df['Age'] = train_df['Age'].round().astype(int)

test_df['Age'] = test_df['Age'].round().astype(int)

# Convert Fare to int

train_df['Fare'] = train_df['Fare'].round().astype(int)

test_df['Fare'] = test_df['Fare'].round().astype(int)"""
# Make the data set

#x = train_df.drop('Survived',axis=1)

#y = train_df['Survived']

x = train_df

y = train_y
# Show what the dtype of x, note that the catboost will just make the string object to categorical object inside

x.dtypes
# Choose the features we want to train, just forget the float data

select_features_index = np.where(x.dtypes != float)[0]

select_features_index
# Make the x for train & validation

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.85,random_state=1234)
x_train.head()
y_train.head()
# Make the catboost model, use_best_model params will make the model prevent overfitting

model = CatBoostClassifier(eval_metric='Accuracy', use_best_model=True,random_seed=42)
# Fit model to the data

model.fit(x_train,y_train,cat_features=select_features_index, eval_set=(x_test,y_test))
# Since the data is not so big, it is better to use CV for the model, use 10 folds

params = model.get_params()

params['loss_function']='Logloss'

cv_data = cv(Pool(x,y,cat_features=select_features_index),params,fold_count=10)
# Print all the columns in the cv_data df

cv_data.columns.values
# show the accuracy of the model

print ('best cv accuracy: {}'.format(max(cv_data['test-Accuracy-mean'])))
# Show the model test accuracy, but you have to note that the acc is not the cv acc, so recommend to use the cv acc to evaluate model!

print('test accuracy: {:.6f}'.format(accuracy_score(y_test,model.predict(x_test))))
# Make the submission file, make sure to convert pred to int

pred = model.predict(test_df)

pred = pred.astype(int)

submission = pd.DataFrame({'PassengerId':test_id,'Survived':pred})
submission.head()
# Save the file to directory

submission.to_csv('catboost1.csv',index=False)
y_pred = submission['Survived'].values

y_true = answer['Survived'].values

accuracy_score(y_true,y_pred)
train_df = pd.read_pickle('../input/titanic-top-3-full-eda-model/train_x.pkl')

train_y = pd.read_pickle('../input/titanic-top-3-full-eda-model/train_y.pkl')

test_df = pd.read_pickle('../input/titanic-top-3-full-eda-model/test_x.pkl')
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



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
def features_hist(df1,df2,features,height):

    fig, ax = plt.subplots(nrows=len(features),ncols=1,figsize=(20, height*len(features)))

    for i,f in enumerate(features):

        df1.hist(column=f,bins=40,ax=ax[i])

        if df2.__class__.__name__ == 'DataFrame':

            df2.hist(column=f,bins=40,ax=ax[i])

        ax[i].set_title(f)    
features_hist(train_df,test_df,list(train_df.columns.values),5)
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer

def norm_scaler(df,features,covariates,method='standard',scaler=None):

    if method=='standard' or method=='power2':

        df_transform = df[features].to_dict(orient='list')

        df_out = []        

        for f in features:

            df_out.append(df_transform[f])

        if scaler == None:

            if method =='standard':

                scaler = StandardScaler()

            elif method == 'power2':

                scaler = PowerTransformer()

            scaler.fit(np.array(df_out).T)

        df_out = scaler.transform(np.array(df_out).T)

        df_out = pd.DataFrame(df_out,columns=features)

    elif method == 'log':

        df_out = df[features]

        df_out = df_out.applymap(lambda x: np.log(1+x))

    elif method == 'power1':

        df_out = df[features]

        df_out = df_out.applymap(lambda x: np.sqrt(x+(2/3)))  

    for cov in covariates:

        df_out[cov] = list(df[cov])

    return df_out, scaler       
df_all = pd.concat([train_df,test_df])

df_all_n,scaler1 = norm_scaler(df_all,list(train_df.columns.values),[])

train_df_n,scaler = norm_scaler(train_df,list(train_df.columns.values),[],'standard',scaler1)

test_df_n,scaler = norm_scaler(test_df,list(train_df.columns.values),[],'standard',scaler1)
features_hist(train_df_n,test_df_n,list(train_df.columns.values),5)
train_df_n.describe()
test_df_n.describe()
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
# split dataset in cross-validation

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



# create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



# create table to compare MLA predictions

MLA_predict = train_y
train_y = pd.read_pickle('../input/titanic-top-3-full-eda-model/train_y.pkl')
train_y.shape
# index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation

    cv_results = model_selection.cross_validate(alg, train_df_n, train_y, cv=cv_split, return_train_score=True)



    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    #save MLA predictions - see section 6 for usage

    alg.fit(train_df_n, train_y)

    MLA_predict[MLA_name] = alg.predict(train_df_n)

    

    row_index+=1



# print and sort table

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare

# MLA_predict
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
def plot_feature_importances(model, columns):

    nr_f = 11

    imp = pd.Series(data = model.best_estimator_.feature_importances_, 

                    index=columns).sort_values(ascending=False)

    plt.figure(figsize=(7,5))

    plt.title("Feature importance")

    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')
# Baseline model

dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, train_df, train_y, cv=cv_split, return_train_score=True)

dtree.fit(train_df, train_y)



print('BEFORE DT Parameters: ', dtree.get_params())

print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

#print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))

print('-'*10)
param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini

              #'splitter': ['best','random'], #splitting methodology; two supported strategies - default is best

              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none

              'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2

              'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1

              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all

              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

             }



#choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train_df, train_y)



#print(tune_model.cv_results_.keys())

#print(tune_model.cv_results_['params'])

print('AFTER DT Parameters: ', tune_model.best_params_)

#print(tune_model.cv_results_['mean_train_score'])

print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

#print(tune_model.cv_results_['mean_test_score'])

print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
plot_feature_importances(tune_model,train_df.columns)
#selected_features = ['Title','Pclass','haveCabin','BigFamily','FareBin']

selected_features = list(train_df.columns.values)
#base model

print('BEFORE DT RFE Training Shape Old: ', train_df.shape) 

print('BEFORE DT RFE Training Columns Old: ', train_df.columns.values)



print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 

print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print('-'*10)





#feature selection

dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)

dtree_rfe.fit(train_df_n, train_y)



#transform x&y to reduced features and fit new model

#alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

X_rfe = train_df.columns.values[dtree_rfe.get_support()]

rfe_results = model_selection.cross_validate(dtree, train_df[X_rfe], train_y, cv  = cv_split, return_train_score=True)



#print(dtree_rfe.grid_scores_)

print('AFTER DT RFE Training Shape New: ', train_df[X_rfe].shape) 

print('AFTER DT RFE Training Columns New: ', X_rfe)



print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 

print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))

print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))

print('-'*10)





#tune rfe model

rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

rfe_tune_model.fit(train_df[selected_features], train_y)



#print(rfe_tune_model.cv_results_.keys())

#print(rfe_tune_model.cv_results_['params'])

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)

#print(rfe_tune_model.cv_results_['mean_train_score'])

print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

#print(rfe_tune_model.cv_results_['mean_test_score'])

print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
#Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

import graphviz 

dot_data = tree.export_graphviz(dtree, out_file=None, 

                                feature_names = list(train_df.columns.values), class_names = True,

                                filled = True, rounded = True)

graph = graphviz.Source(dot_data) 

graph
def correlation_heatmap(df,absolute):

    _ , ax = plt.subplots(figsize =(16, 16))

    

    if absolute:

        corr = df.corr().abs()

        colormap = sns.color_palette("Reds")

    else:

        corr = df.corr()

        colormap = sns.diverging_palette(220, 10, as_cmap = True)

        

    _ = sns.heatmap(

        corr, 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':10 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
MLA_predict_values = np.array([x.tolist() for x in MLA_predict[891:].values.tolist()]).T

MLA_predict2 = pd.DataFrame(data = MLA_predict_values,columns = list(MLA_predict[891:].index))

MLA_predict2.insert(loc=0,column='Survived',value=train_y)

correlation_heatmap(MLA_predict2,False)
#why choose one model, when you can pick them all with voting classifier

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

#removed models w/o attribute 'predict_proba' required for vote classifier and models with a 1.0 correlation to another model

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





# Hard Vote or majority rules

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

vote_hard_cv = model_selection.cross_validate(vote_hard,train_df_n[selected_features],train_y,cv=cv_split,return_train_score=True)

vote_hard.fit(train_df_n[selected_features],train_y)



print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 

print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

print('-'*10)





# Soft Vote or weighted probabilities

vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

vote_soft_cv = model_selection.cross_validate(vote_soft,train_df_n[selected_features],train_y,cv=cv_split,return_train_score=True)

vote_soft.fit(train_df_n[selected_features],train_y)



print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 

print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))

print('-'*10)
import time



#WARNING: Running is very computational intensive and time expensive.

#Code is written for experimental/developmental purposes and not production ready!





#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

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

            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

            'n_estimators': grid_n_estimator, #default=50

            'learning_rate': grid_learn, #default=1

            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R

            'random_state': grid_seed

            }],

       

    

            [{

            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

            'n_estimators': grid_n_estimator, #default=10

            'max_samples': grid_ratio, #default=1.0

            'random_state': grid_seed

             }],



    

            [{

            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'random_state': grid_seed

             }],





            [{

            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

            #'loss': ['deviance', 'exponential'], #default=’deviance’

            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.

            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”

            'max_depth': grid_max_depth, #default=3   

            'random_state': grid_seed

             }],



    

            [{

            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.

            'random_state': grid_seed

             }],

    

            [{    

            #GaussianProcessClassifier

            'max_iter_predict': grid_n_estimator, #default: 100

            'random_state': grid_seed

            }],

        

    

            [{

            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

            'fit_intercept': grid_bool, #default: True

            #'penalty': ['l1','l2'],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs

            'random_state': grid_seed

             }],

            

    

            [{

            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

            'alpha': grid_ratio, #default: 1.0

             }],

    

    

            #GaussianNB - 

            [{}],

    

            [{

            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

            'n_neighbors': [1,2,3,4,5,6,7], #default: 5

            'weights': ['uniform', 'distance'], #default = ‘uniform’

            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

            }],

            

    

            [{

            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r

            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

            'C': [1,2,3,4,5], #default=1.0

            'gamma': grid_ratio, #edfault: auto

            'decision_function_shape': ['ovo', 'ovr'], #default:ovr

            'probability': [True],

            'random_state': grid_seed

             }],



    

            [{

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': grid_learn, #default: .3

            'max_depth': [1,2,4,6,8,10], #default 2

            'n_estimators': grid_n_estimator, 

            'seed': grid_seed  

             }]   

        ]





# create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD','MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



# index through MLA and save performance to table

row_index = 0



start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip



    MLA_name = clf[1].__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(clf[1].get_params())

    

    start = time.perf_counter()        

    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv=cv_split, scoring = 'roc_auc', return_train_score=True)

    best_search.fit(train_df_n[selected_features],train_y)

    run = time.perf_counter() - start

    

    MLA_compare.loc[row_index, 'MLA Time'] = best_search.cv_results_['mean_fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = best_search.cv_results_['mean_train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = best_search.cv_results_['mean_test_score'].mean()   

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = best_search.cv_results_['mean_test_score'].std()*3 



    best_param = best_search.best_params_

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))

    clf[1].set_params(**best_param) 

    

    row_index += 1

    

run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))



print('-'*10)
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard,train_df_n[selected_features],train_y,cv=cv_split,return_train_score=True)

grid_hard.fit(train_df_n[selected_features],train_y)



print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 

print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))

print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))

print('-'*10)



row_index = 12

MLA_compare.loc[row_index, 'MLA Name'] = 'Hard Voting'

MLA_compare.loc[row_index, 'MLA Parameters'] = str(grid_hard.get_params())

MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = grid_hard_cv['train_score'].mean()

MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = grid_hard_cv['test_score'].mean()   

MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = grid_hard_cv['test_score'].std()*3 

MLA_compare.loc[row_index, 'MLA Time'] = grid_hard_cv['fit_time'].mean()



#Soft Vote or weighted probabilities w/Tuned Hyperparameters

grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft,train_df_n[selected_features],train_y,cv=cv_split,return_train_score=True)

grid_soft.fit(train_df_n[selected_features],train_y)



print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 

print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))

print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))

print('-'*10)



row_index = 13

MLA_compare.loc[row_index, 'MLA Name'] = 'Soft Voting'

MLA_compare.loc[row_index, 'MLA Parameters'] = str(grid_soft.get_params())

MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = grid_soft_cv['train_score'].mean()

MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = grid_soft_cv['test_score'].mean()   

MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = grid_soft_cv['test_score'].std()*3 

MLA_compare.loc[row_index, 'MLA Time'] = grid_soft_cv['fit_time'].mean()
MLA_compare
def get_best_score(model):

    print(model.best_score_)    

    print(model.best_params_)

    print(model.best_estimator_)

    return model.best_score_



def plot_feature_importances(model, columns):

    nr_f = 10

    imp = pd.Series(data = model.best_estimator_.feature_importances_, 

                    index=columns).sort_values(ascending=False)

    plt.figure(figsize=(7,5))

    plt.title("Feature importance")

    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')



def submission(test_set,df_test,model_name,model,best_score,cv_scores):

    cv_scores[model_name] = best_score

    pred = model.predict(test_set)

    submission = pd.DataFrame()

    submission['PassengerId'] = df_test['PassengerId']

    submission['Survived'] =pred

    submission.to_csv('%s.csv' %model_name,index=False)

    return submission



def accuracy(submission,answer):

    y_pred = submission['Survived'].values

    y_true = answer['Survived'].values

    return accuracy_score(y_true,y_pred)
start_total = time.perf_counter()



for clf, param in zip (vote_est, grid_param):

    print (clf[1].__class__.__name__)

    start = time.perf_counter()        

    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv=cv_split, scoring = 'roc_auc', return_train_score=True)

    best_search.fit(train_df_n[selected_features],train_y)

    run = time.perf_counter() - start

    print('The best parameter is {} with a runtime of {:.2f} seconds.'.format(best_param, run))

    

    pred = best_search.predict(test_df_n[selected_features])

    submission = pd.DataFrame()

    submission['PassengerId'] = test_id

    submission['Survived'] =pred

    submission.to_csv('%s.csv' %clf[0],index=False)

    

    print ('Test Accuracy:')

    print (accuracy(submission,answer))

    

run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))
print ('Hard vote')

pred = grid_hard.predict(test_df_n[selected_features])

submission = pd.DataFrame()

submission['PassengerId'] = test_id

submission['Survived'] =pred

submission.to_csv('hard_vote.csv',index=False)



print ('Test Accuracy:')

print (accuracy(submission,answer))



print ('Soft vote')

pred = grid_soft.predict(test_df_n[selected_features])

submission = pd.DataFrame()

submission['PassengerId'] = test_id

submission['Survived'] =pred

submission.to_csv('hard_vote.csv',index=False)



print ('Test Accuracy:')

print (accuracy(submission,answer))