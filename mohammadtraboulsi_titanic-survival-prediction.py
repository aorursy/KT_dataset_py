

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import sklearn as skl

import random

import time

from IPython import display 



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# our data

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



# create copies



df1 = df_train.copy(deep = True)

df2 = df_test.copy(deep =True)



# create reference to clean when needed



df_clean = [df1,df2]



# preview data



df1.head()
print('Missing values in the training dataset:')

print (df1.isnull().sum())

print()

print ('Missing values in the Test dataset:')

print(df2.isnull().sum())
# remove unnecessary columns

dropped_cols = ['PassengerId','Ticket','Cabin']

df1.drop(dropped_cols, axis=1,inplace=True)

df2.drop(dropped_cols, axis=1,inplace=True)
# fill missing values in both datasets



for df in df_clean:

    # numbers are filled with the median

    df['Age'].fillna(df['Age'].median(),inplace=True)

    df['Fare'].fillna(df['Fare'].median(),inplace=True)

    # filled by mode since it has categorical choices (S class)

    df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
# recheck values again

print('Missing values in the training dataset:')

print (df1.isnull().sum() == 0)

print()

print ('Missing values in the Test dataset:')

print(df2.isnull().sum()== 0)
df1.head( )
#sibsp and parch can be used to compute family size:



for df in df_clean:

    df['Family'] = df['SibSp'] + df['Parch'] + 1

    df['Alone'] = 1

    df['Alone'].loc[df['Family'] > 1 ] = 0 # if family is greater than 1 then they are not alone

# extract titles from names:

# certain titles are replaced because they are similar

replace_titles = {"Mlle": "Miss","Ms": "Miss",'Mme':'Miss' }

for df in df_clean:

    df['Title'] = ''

    df['Title'] = df['Name'].str.split(', ',expand=True)[1].str.split('.',expand=True)

    df.replace({'Title':replace_titles},inplace=True)

    rare_titles = df['Title'].value_counts() <10  # titles that are not statistically significant will be grouped together

    df['Title'] =df['Title'].apply(lambda x: 'Other' if rare_titles.loc[x] == True else x)



    df['FareBin'] = pd.qcut (df['Fare'],4) # cut according to equal frequencies 

    df['AgeBin'] = pd.cut(df['Age'].astype('int'),5) # cut into equal intervals

print(df1['Title'].value_counts())



print('_'*15)

print(df1.info())

print('_'*15)

print(df2.info())

print('_'*15)

df1.head()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



label = LabelEncoder()



label_cols = ['Sex','Embarked','Title','AgeBin','FareBin']



for df in df_clean:

    for col in label_cols:

        df[col +'_Coded'] = label.fit_transform(df[col])
y = ['Survived']

x_pretty = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'Family', 'Alone'] # main features used when printing 

x_coded = ['Sex_Coded','Pclass', 'Embarked_Coded', 'Title_Coded','SibSp', 'Parch', 'Age', 'Fare'] # used for calculations

x_bin = ['Sex_Coded','Pclass', 'Embarked_Coded', 'Title_Coded', 'Family', 'AgeBin_Coded', 'FareBin_Coded'] # continuous variables and bins

xy = y + x_pretty 

xy_bin = y+x_bin



df_x_dummy = pd.get_dummies(df1[x_pretty])

x_dummy = df_x_dummy.columns.tolist()

xy_dummy = y+ x_dummy 





print('Original Columns:',xy)



print('Bins: ',xy_bin)

print('Dummy: ',xy_dummy)



df_x_dummy.head()
from sklearn import model_selection



# we are using columns in 3 different formats



train_x, test_x, train_y, test_y = model_selection.train_test_split(df1[x_coded],df1[y])

train_x_bin, test_x_bin, train_y_bin, test_y_bin = model_selection.train_test_split(df1[x_bin],df1[y])
# correlations



for x in x_pretty:

    if df1[x].dtype != 'float64': 

        print('Survival Correlation by:', x)

        print(df1[[x, y[0]]].groupby(x,as_index=False).mean())

        print('_'*15)
plt.figure(figsize=(15,6))



plt.subplot(131)



plt.boxplot(x=df1['Fare'],meanline=True,showmeans=True)

plt.title('Fare Boxplot')



plt.subplot(132)

plt.boxplot(x=df1['Age'],meanline=True,showmeans=True)

plt.title('Age Boxplot')



plt.subplot(133)

plt.boxplot(x=df1['Family'],meanline=True,showmeans=True)

plt.title('Family Boxplot');
# survival rate 

from operator import attrgetter

plt.figure(figsize=(15,6))



plt.subplot(131)

survival_rate = df1.groupby('FareBin',as_index=False).mean()['Survived']

xtick_labels = ['very low','low','medium','high']

plt.bar(range(4),survival_rate)

plt.xticks(range(4),xtick_labels)

plt.xlabel('Fare')

plt.title('Fare Survival Rare (survived/not survived)')



plt.subplot(132)

group1 = df1.groupby('AgeBin',as_index=False).mean()

survival_rate = group1['Survived']

xtick_labels = np.insert(group1['AgeBin'].map(attrgetter('right')).tolist(),0,0)

plt.bar(x=range(6),height=np.insert(survival_rate.values,0,0),align='edge',width=-1,tick_label=xtick_labels)

plt.title('Survival Rate By Age Group')

plt.xlim(0,5)



plt.subplot(133)

survival_rate = df1.groupby('Family',as_index=False).mean()[['Family','Survived']]

plt.bar(survival_rate['Family'],survival_rate['Survived'])

plt.xticks(range(1,12))

plt.title('Survival Rate By Family Size (1==alone)');
fig, saxis = plt.subplots(1,3,figsize=(15, 6))



sns.barplot(data=df1,x='Embarked',y='Survived',ax=saxis[0],hue='Sex')

sns.barplot(data=df1,x='Pclass',y='Survived',ax=saxis[1],hue='Sex')

sns.barplot(data=df1,x='Alone',y='Survived',ax=saxis[2],hue='Sex');
# Pclass vs other variables



fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,6))



sns.boxplot(x=df1['Pclass'],y=df1['Fare'],hue=df1['Survived'],ax=ax1)

ax1.set_title('Pclass vs Fare Survival')



sns.boxplot(data=df1,x='Pclass',y='Age',hue='Survived',ax=ax2)

ax2.set_title('Pclass vs Age Survival')



sns.violinplot(data=df1,x='Pclass',y='Family',hue='Survived',ax=ax3,split=True)

ax3.set_title('Pclass vs Family Size Survival');
plt.figure(figsize=(15,6))

plt.subplot(121)



sns.pointplot(data=df1,x='Family',y='Survived',hue='Sex')

plt.title('Family Size vs Gender Survival')



plt.subplot(122)

sns.pointplot(data=df1,x='Pclass',y='Survived',hue='Sex')

plt.title('Pclass vs Gender Survival');
g = sns.FacetGrid(data=df1,col='Embarked',height=4)

g.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')

g.add_legend();
#heatmap

plt.figure(figsize=(20,8))

sns.heatmap(df1.corr(),annot=True,cmap=sns.diverging_palette(10, 220, sep=80, n=50))

plt.title('Pearson Correlation Heatmap of Features');
# models with default parameters

from sklearn import ensemble, svm ,tree, naive_bayes,neighbors,linear_model, discriminant_analysis, gaussian_process

from sklearn import feature_selection, model_selection, metrics

from xgboost import XGBClassifier



MLA =[

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    

    gaussian_process.GaussianProcessClassifier(),

    

    

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    neighbors.KNeighborsClassifier(),

    

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    

    XGBClassifier()

    

]





# split training data 



cv_split= model_selection.ShuffleSplit(n_splits=10,test_size=0.3,train_size=0.6,random_state=42)



MLA_cols = ['Name','Params', 'Train Accuracy (mean)', 'Test Accuracy (mean)', 'Test_STD_3', 'Time']

MLA_df = pd.DataFrame(columns = MLA_cols)

MLA_predict = df1[y]



for row_nb, alg in enumerate(MLA):

    

    MLA_df.loc[row_nb, 'Name'] = alg.__class__.__name__ # algorithm name

    cv_result = model_selection.cross_validate(alg,df1[x_bin],df1[y],cv=cv_split,return_train_score=True)

    MLA_df.loc[row_nb,'Params'] = str(alg.get_params())

    MLA_df.loc[row_nb,'Time'] = cv_result['fit_time'].mean()

    MLA_df.loc[row_nb,'Train Accuracy (mean)'] = cv_result['train_score'].mean()

    MLA_df.loc[row_nb,'Test Accuracy (mean)'] = cv_result['test_score'].mean()

    MLA_df.loc[row_nb,'Test_STD_3'] = cv_result['test_score'].std() * 3

  

    alg.fit(df1[x_bin],df1[y].values.ravel())

    MLA_predict[alg.__class__.__name__] =alg.predict(df1[x_bin])



MLA_df.sort_values(by='Test Accuracy (mean)',inplace=True,ascending=False)

MLA_df

    
# Confusion Matrix Example

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



y_pred = cross_val_predict(svm.SVC(probability=True),df1[x_bin],df1[y],cv=10)

sns.heatmap(confusion_matrix(df1[y],y_pred,normalize='true'),annot=True)

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title('Confusion Matrix For SVC classifier');

# Tuning parameters for SVC classifier





model = svm.SVC(probability=True)

base_results = model_selection.cross_validate(model, df1[x_bin],df1[y],cv=cv_split)

model.fit(df1[x_bin],df1[y])



print ('Default params:',model.get_params())

print('Test Score With Default Params:',base_results['test_score'].mean()*100 )

print('STD *3 with Default Params:', base_results['test_score'].std() *300)



print('-'*25)





param_grid = {

    'kernel': ['rbf'], # kernel parameters selects the type of hyperplane used to separate the data.

    'gamma': [0.01], # gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set

    'C': [1], # C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.

    'degree': [2], #degree is a parameter used when kernel is set to ‘poly’. It’s basically the degree of the polynomial used to find the hyperplane to split the data.

}



tune_model = model_selection.GridSearchCV(model,param_grid=param_grid,cv=cv_split,scoring='roc_auc')

tune_model.fit(df1[x_bin],df1[y])



print ('Best params:',tune_model.best_params_)

print('Test Score With Default Params:',tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100 )

print('STD *3 with Default Params:', tune_model.cv_results_['std_test_score'][tune_model.best_index_]*300)

# Tuning parameters for RandomForest classifier





model = ensemble.RandomForestClassifier()

base_results = model_selection.cross_validate(model, df1[x_bin],df1[y],cv=cv_split)

model.fit(df1[x_bin],df1[y])



print ('Default params:',model.get_params())

print('Test Score With Default Params:',base_results['test_score'].mean()*100 )

print('STD *3 with Default Params:', base_results['test_score'].std() *300)



print('-'*25)





param_grid = {

    'n_estimators' : [100],

    'max_features' : ['auto'],

    'max_depth': [5],

    'min_samples_split': [2],

    'min_samples_leaf' : [2],

    'bootstrap': [False],

}



tune_model = model_selection.GridSearchCV(model,param_grid=param_grid,cv=cv_split,scoring='roc_auc')

tune_model.fit(df1[x_bin],df1[y])



print ('Best params:',tune_model.best_params_)

print('Test Score With Default Params:',tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100 )

print('STD *3 with Default Params:', tune_model.cv_results_['std_test_score'][tune_model.best_index_]*300)

# base model



print('Data Default Shape: ',df1[x_bin].shape)

print('Default Features: ',df1[x_bin].columns.values)



print('For model: ',model.__class__.__name__)



print('Test Score With Default Params:',base_results['test_score'].mean()*100 )

print('STD *3 with Default Params:', base_results['test_score'].std() *300)



# feature selection



feature_model = feature_selection.RFECV(model, step=1,scoring='accuracy',cv=cv_split,n_jobs=-1)

feature_model.fit(df1[x_bin],df1[y])



# transform to reduced features and fit model



x_rfe = df1[x_bin].columns.values[feature_model.get_support()]

x_rfe_results = model_selection.cross_validate(model,df1[x_rfe],df1[y],cv=cv_split)





print('_'*25)

print('Data Current Shape: ',df1[x_rfe].shape)

print('Current Features: ',df1[x_rfe].columns.values)

print('Test Score With Current Params:',x_rfe_results['test_score'].mean() )

print('STD *3 with Default Params:', x_rfe_results['test_score'].std() *300)



# tune rfe model

print('-'*25)
param_grid = {

    'criterion': ['entropy'],

    'n_estimators' : [1000],

    'max_features' : ['auto'],

    'max_depth': [4],

    'min_samples_split': [10],

    'min_samples_leaf' : [1],

    'bootstrap': [False],

}



tune_model = model_selection.GridSearchCV(model,param_grid=param_grid,cv=cv_split,scoring='roc_auc')

tune_model.fit(df1[x_rfe],df1[y])



print ('Best params:',tune_model.best_params_)

print('Test Score With Default Params:',tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100 )

print('STD *3 with Default Params:', tune_model.cv_results_['std_test_score'][tune_model.best_index_]*300)
# show wrong predictions



predictions = tune_model.predict(df1[x_rfe])

df_error = pd.DataFrame(columns = df1[x_rfe].columns)

df_error['idx'] =1

df_error['pred'] = 1

df_error['real'] =1

cnt =0 

for i in range(891):

    if predictions[i] != df1[y].loc[i][0]:

        df_error.loc[cnt] = df1[x_rfe].loc[i]

        df_error['idx'].loc[cnt] = i

        df_error['pred'].loc[cnt] = predictions[i]

        df_error['real'].loc[cnt] = df1[y].loc[i][0]

        cnt +=1

df_error.Sex_Coded.value_counts().plot(kind='pie')

plt.show()

df_error.Family.value_counts().plot(kind='bar')

plt.show()



df_error
plt.figure(figsize=(20,8))

sns.heatmap(MLA_predict.corr(),annot=True)
# voting classifier 

# remove all similar estimators first (those with correlation close to 1)



models = [

    

    ('ada',ensemble.AdaBoostClassifier()),

    ('bc', ensemble.BaggingClassifier()),

#     ('etc',ensemble.ExtraTreesClassifier()),

    ('gbc', ensemble.GradientBoostingClassifier()),

    ('rfc', ensemble.RandomForestClassifier()),

    

    ('gpc', gaussian_process.GaussianProcessClassifier()),

    

    ('lr', linear_model.LogisticRegressionCV()),

    

    ('bnb', naive_bayes.BernoulliNB()),

    ('gnb', naive_bayes.GaussianNB()),

    

    ('knn', neighbors.KNeighborsClassifier()),

    

    ('svc', svm.SVC(probability=True)),

    

    ('xgb', XGBClassifier())



]



# Hard Vote (majority rules)



vote_hard = ensemble.VotingClassifier(estimators=models,voting='hard')



vote_hard_cv = model_selection.cross_validate(vote_hard,df1[x_rfe],df1[y],cv=cv_split)

vote_hard.fit(df1[x_rfe],df1[y])



print('Hard Voting:')



print('Test accuracy: ', vote_hard_cv['test_score'].mean() *100 )

print('Test STD *3 +/-: ', vote_hard_cv['test_score'].std() *300)



print('-'*25)



# soft voting (weighted average)





vote_soft = ensemble.VotingClassifier(estimators=models,voting='soft')



vote_soft_cv = model_selection.cross_validate(vote_hard,df1[x_rfe],df1[y],cv=cv_split)

vote_soft.fit(df1[x_rfe],df1[y])



print('Soft Voting:')



print('Test accuracy: ', vote_soft_cv['test_score'].mean() *100 )

print('Test STD *3 +/-: ', vote_soft_cv['test_score'].std() *300)
# Hyperparameter Tune

n_estimators = [10,50,100,300]

ratio = [.1, .25, .5, .75, 1.0]

learn = [.01, .03, .05, .1, .25]

max_depth = [2, 4, 6, 8, 10, None]

min_samples = [5, 10, .03, .05, .10]

criterion = ['gini', 'entropy']

bools = [True, False]





grid_params = [

    [{

        # adaboost params

        'n_estimators': [300],

        'learning_rate': [0.1],

        

        

    }],

    

    [{

        # bagging classifier

        'n_estimators': [300],

        'max_samples': [0.1]

    }],

    

#     [{

#         # Extra Tree Classifier

#         'n_estimators': n_estimators, #default=10

#             'criterion': criterion, #default=”gini”

#             'max_depth': max_depth, #default=None

#     }],

    

    [{

        #gradient boost classifier

        

        'learning_rate': [.05],

        'n_estimators': [300],

        'max_depth': [2],

    }],

    

    [{

        # RandomForestClassifier 

        'n_estimators': [100],

         'criterion': ['entropy'], #default=”gini”

        'max_depth': [6], #default=None

    }],

    

    [{

    # gaussian classifier

        

        'max_iter_predict': [10]

    }],

    

    [{

        #linearregressioncv

        'fit_intercept': [True],

        'solver': [ 'sag']

    }],

    

    [{

        #bernoulli

        

    }],

    

    [{

        #gaussiannb

    }],

    

    [{

        #knn

        'n_neighbors': [7], #default: 5

        'weights': ['uniform'], #default = ‘uniform’

        'algorithm': ['brute']

    }],

    

    [{

    'kernel': ['rbf'], # kernel parameters selects the type of hyperplane used to separate the data.

    'gamma': [0.01], # gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set

    'C': [1], # C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.

    'degree': [2], #degree is a parameter used when kernel is set to ‘poly’. It’s basically the degree of the polynomial used to find the hyperplane to split the data.

        

    }],

    

    [{

        'learning_rate': [0.1],

        'max_depth': [6],

        'n_estimators': [10]

    }]

    

]



for est, param in zip(models, grid_params):

    

    best_modelcv = model_selection.RandomizedSearchCV(estimator= est[1],param_distributions=param,cv=cv_split,scoring='roc_auc')

    best_modelcv.fit(df1[x_rfe],df1[y])

    print('Best params for {} is {}, test_accuracy: {}'.format(est[1].__class__.__name__, best_modelcv.best_params_,best_modelcv.cv_results_['mean_test_score'][best_modelcv.best_index_] *100))

    

    est[1].set_params(**best_modelcv.best_params_)

    

    

# Best params for AdaBoostClassifier is {'n_estimators': 300, 'learning_rate': 0.1}, test_accuracy: 87.86301576600705

# Best params for BaggingClassifier is {'n_estimators': 300, 'max_samples': 0.1}, test_accuracy: 87.70164012872436

# Best params for GradientBoostingClassifier is {'n_estimators': 300, 'max_depth': 2, 'learning_rate': 0.05}, test_accuracy: 88.89252539733737

# Best params for RandomForestClassifier is {'n_estimators': 100, 'max_depth': 6, 'criterion': 'entropy'}, test_accuracy: 88.17643055916224

# Best params for GaussianProcessClassifier is {'max_iter_predict': 10}, test_accuracy: 87.10541901964929

# Best params for LogisticRegressionCV is {'solver': 'newton-cg', 'fit_intercept': True}, test_accuracy: 86.34680076314885

# Best params for BernoulliNB is {}, test_accuracy: 81.09156034962865

# Best params for GaussianNB is {}, test_accuracy: 86.7309713272034

# Best params for KNeighborsClassifier is {'weights': 'uniform', 'n_neighbors': 7, 'algorithm': 'brute'}, test_accuracy: 85.4860481775351

# Best params for SVC is {'kernel': 'rbf', 'gamma': 0.01, 'degree': 2, 'C': 1}, test_accuracy: 86.78288856249846

# Best params for XGBClassifier is {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}, test_accuracy: 88.91386910048607
## Same steps but now with tuned params



# Hard Vote (majority rules)



vote_hard = ensemble.VotingClassifier(estimators=models,voting='hard')



vote_hard_cv = model_selection.cross_validate(vote_hard,df1[x_rfe],df1[y],cv=cv_split)

vote_hard.fit(df1[x_rfe],df1[y])



print('Hard Voting:')



print('Test accuracy: ', vote_hard_cv['test_score'].mean() *100 )

print('Test STD *3 +/-: ', vote_hard_cv['test_score'].std() *300)



print('-'*25)



# soft voting (weighted average)





vote_soft = ensemble.VotingClassifier(estimators=models,voting='soft')



vote_soft_cv = model_selection.cross_validate(vote_soft,df1[x_rfe],df1[y],cv=cv_split)

vote_soft.fit(df1[x_rfe],df1[y])



print('Soft Voting:')



print('Test accuracy: ', vote_soft_cv['test_score'].mean() *100 )

print('Test STD *3 +/-: ', vote_soft_cv['test_score'].std() *300)
# model = models[-1][1]

# model.fit(df1[x_rfe],df1[y])

# df_test['Survived'] = models[-1][1].predict(df2[x_rfe])

# submit = df_test[['PassengerId','Survived']]





# submit.to_csv('../working/submit.csv',index=False)

# if you submit this cell the public score is around 0.77
dropped_cols = ['Sex_male','Embarked_S','Title_Other']

df_x_dummy.drop(dropped_cols,axis=1,inplace=True)
df_x_dummy[['Age','Fare']] =(df_x_dummy[['Age','Fare']]-df_x_dummy[['Age','Fare']].min())/(df_x_dummy[['Age','Fare']].max()-df_x_dummy[['Age','Fare']].min())

df_x_dummy



# df_x_dummy['Fare'] = min_max_scaler.fit_transform(df_x_dummy['Fare'] ) 
train_x_dummy, test_x_dummy, train_y_dummy, test_y_dummy = model_selection.train_test_split(df_x_dummy[[x for x in x_dummy if x not in dropped_cols]],df1[y])
# # important modules needed for GP

# import pickle

# import operator

# import math

# import string

# import random



# from deap import algorithms, base, creator, tools, gp

# nb_iterations = 1 # Warning Can be very computationally expensive. Set to high values  only if you can wait.





# def protectedDiv(left, right):

#     '''

#     This is a function that replaces normal division to prevent zero division errors 

#     '''

#     try:

#         return left / right

#     except ZeroDivisionError:

#             return left





#     # next step is selecting primitives (operators such as addition, multiplication etc)



# pset = gp.PrimitiveSet('Main', 14)

# pset.addPrimitive(operator.add,2) #  addprimitive takes 2 arguments: the operator and number of operations (e.g you add 2 numbers)

# pset.addPrimitive(operator.sub, 2)

# pset.addPrimitive(operator.mul, 2)

# pset.addPrimitive(protectedDiv,2) # note we use protected div instead of normal div

# pset.addPrimitive(math.cos, 1)

# pset.addPrimitive(math.sin, 1)

# pset.addPrimitive(max, 2)

# pset.addPrimitive(min, 2)

# pset.addPrimitive(math.tanh,1)

# pset.addPrimitive(abs,1)







# pset.renameArguments(ARG0='x1')

# pset.renameArguments(ARG1='x2')

# pset.renameArguments(ARG2='x3')

# pset.renameArguments(ARG3='x4')

# pset.renameArguments(ARG4='x5')

# pset.renameArguments(ARG5='x6')

# pset.renameArguments(ARG6='x7')

# pset.renameArguments(ARG7='x8')

# pset.renameArguments(ARG8='x9')

# pset.renameArguments(ARG9='x10')

# pset.renameArguments(ARG10='x11')

# pset.renameArguments(ARG11='x12')

# pset.renameArguments(ARG12='x13')

# pset.renameArguments(ARG13='x14')

# pset.renameArguments(ARG14='x15')

# pset.renameArguments(ARG15='x16')

# pset.renameArguments(ARG16='x17')

# pset.renameArguments(ARG17='x18')

# pset.renameArguments(ARG18='x19')

# pset.renameArguments(ARG19='x20')

# pset.renameArguments(ARG20='x21')

# pset.renameArguments(ARG21='x22')

# pset.renameArguments(ARG22='x23')

# pset.renameArguments(ARG23='x24')

# pset.renameArguments(ARG24='x25')

# pset.renameArguments(ARG25='x26')

# pset.renameArguments(ARG26='x27')

# pset.renameArguments(ARG27='x28')

# pset.renameArguments(ARG28='x29')

# pset.renameArguments(ARG29='x30')

# pset.renameArguments(ARG30='x31')

# pset.renameArguments(ARG31='x32')

# pset.renameArguments(ARG32='x33')

# pset.renameArguments(ARG33='x34')

# pset.renameArguments(ARG34='x35')

# pset.renameArguments(ARG35='x36')

# pset.renameArguments(ARG36='x37')

# pset.renameArguments(ARG37='x38')

# pset.renameArguments(ARG38='x39')

# pset.renameArguments(ARG39='x40')

# pset.renameArguments(ARG40='x41')

# pset.renameArguments(ARG41='x42')

# pset.renameArguments(ARG42='x43')

# pset.renameArguments(ARG43='x44')

# pset.renameArguments(ARG44='x45')

# pset.renameArguments(ARG45='x46')

# pset.renameArguments(ARG46='x47')

# pset.renameArguments(ARG47='x48')

# pset.renameArguments(ARG48='x49')

# pset.renameArguments(ARG49='x50')



# def randomString(stringLength=10):

#     """Generate a random string of fixed length """

#     letters = string.ascii_lowercase

#     return ''.join(random.choice(letters) for i in range(stringLength))



# pset.addEphemeralConstant(randomString(), lambda: random.uniform(-10,10)) # ephermal constant used to terminate the terminal





# # create fitness and genotype 



# creator.create("FitnessMin", base.Fitness, weights=(1.0,))

# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)



# # we want to register some parameters specific to the evolution process,using the toolbox

# # some code taken from https://www.kaggle.com/guesejustin/91-genetic-algorithms-explained-using-geap



# toolbox = base.Toolbox()

# toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("compile", gp.compile, pset=pset)



# inputs =  train_x_dummy.values.tolist() #df1[x_rfe].values.tolist()

# outputs = train_y_dummy.values.tolist()

# length_input = len(inputs) +1 



# def evalSymbReg(individual):

# # Transform the tree expression in a callable function

#     func = toolbox.compile(expr=individual)

#     # Evaluate the accuracy of individuals // 1|0 == survived

#     return math.fsum(np.round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs)) / length_input,

# toolbox.register("evaluate", evalSymbReg)

# toolbox.register("select", tools.selTournament, tournsize=3)

# toolbox.register("mate", gp.cxOnePoint)

# toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))



# # stats functions 





# stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

# stats_size = tools.Statistics(len)

# mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

# mstats.register("avg", np.mean)

# mstats.register("std", np.std)

# mstats.register("min", np.min)

# mstats.register("max", np.max)





# pop = toolbox.population(n=300)

# hof = tools.HallOfFame(1)



# pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=nb_iterations, stats=mstats,

#                                halloffame=hof, verbose=True)

    

# result = toolbox.compile(expr=hof[0]) 
# from sklearn.metrics import accuracy_score

# def Outputs(data):

#     return np.round(1.-(1./(1.+np.exp(-data))))

# predicted = Outputs(np.array([result(*x) for x in test_x_dummy.values.tolist()]))

# print(accuracy_score(predicted.astype('int'),test_y_dummy.astype('int')))
# # submission with genetic algorithm



# df_test['surv1'] = vote_soft.predict(df2[x_rfe])



# df_test['Survived'] = Outputs(np.array([result(*x) for x in df2[x_rfe].values.tolist()])).astype(int)

 

# print(accuracy_score(df_test['Survived'].astype('int'),df_test['surv1'].astype('int')))



# submit = df_test[['PassengerId','Survived']]





# # with about  200  iterations the 2 algorithms become very similar (accuracy 99.5)

# # if we want better accuracy, we need more than 200 iterations

# # submit.to_csv('../working/submit3.csv',index=False)

# import tensorflow as tf
# model = tf.keras.Sequential([

#     tf.keras.layers.Flatten(input_shape=(14,)),

#     tf.keras.layers.Dense(42,activation=tf.nn.relu),

#     tf.keras.layers.Dense(28,activation=tf.nn.relu),



#     tf.keras.layers.Dense(14,activation=tf.nn.relu),

#     tf.keras.layers.Dense(7,activation=tf.nn.relu),



#     tf.keras.layers.Dense(1,activation=tf.nn.sigmoid),

# ])
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
# history = model.fit(train_x_dummy,train_y_dummy,epochs=5,verbose=False)

# test_loss, test_acc = model.evaluate(test_x_dummy, test_y_dummy)

# test_acc # accuracy is about 82% similar to other models (epochs=50)
# df_test_dummy =  pd.get_dummies(df2[x_pretty])

# df_test_dummy.drop(dropped_cols,axis=1,inplace=True)

# x_final = [x for x in x_dummy if x not in dropped_cols]

# df_test_dummy['Survived'] = model.predict_classes(df_test_dummy)
# df_test_dummy['PassengerId'] = df_test['PassengerId']

# submit = df_test_dummy[['PassengerId','Survived']]



# submit.to_csv('../working/submit.csv',index=False)
# check our data again

df_feature = pd.get_dummies(df1[x_rfe].astype(str),drop_first=1)

df_feature
train_x_dummy, test_x_dummy, train_y_dummy, test_y_dummy = model_selection.train_test_split(df_feature,df1[y])
# using polynomials 



from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2).fit(train_x_dummy)



train_x_transformed = poly.transform(train_x_dummy)

test_x_transformed = poly.transform(test_x_dummy)
print(poly.get_feature_names())
df_feature
# feature selection

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import cross_val_score



model = XGBClassifier(n_estimators= 10, max_depth= 6, learning_rate = 0.1)

base_results = model_selection.cross_validate(model, df_feature,df1[y],cv=cv_split)

model.fit(df_feature,df1[y])

print('Data Default Shape: ',train_x_transformed.shape)

print('Default Features: ',df_feature.columns.values)



print('For model: ',model.__class__.__name__)



print('Test Score With Default Params:',base_results['test_score'].mean()*100 )

print('STD *3 with Default Params:', base_results['test_score'].std() *300)

highest_score = 0

for i in range(1,train_x_transformed.shape[1] + 1,1):

    select = SelectKBest(score_func=chi2,k=i)

    select.fit(train_x_transformed,train_y_dummy)

    train_x_selected = select.transform(train_x_transformed)

    

    model.fit(train_x_selected,train_y_dummy)

    scores = cross_val_score(model,train_x_selected,train_y_dummy,cv=cv_split)

    print('features: %i, score %.3f +/- %.3f' % (i,np.mean(scores),np.std(scores)))

    

    

    # save feature with highest score

    

    if np.mean(scores) > highest_score:

        highest_score = np.mean(scores)

        std = np.std(scores)

        number_features = i

    elif np.mean(scores) == highest_score:

        if np.std(scores) < std:

            highest_score = np.mean(scores)

            std = np.std(scores)

            number_features = i

print('best number of features  %i,  with score: %.3f +/- %.3f' %(number_features,highest_score,std))
select = SelectKBest(score_func=chi2, k=number_features)

select.fit(train_x_transformed,train_y_dummy)

train_x_selected = select.transform(train_x_transformed)

test_x_selected = select.transform(test_x_transformed)
import tensorflow as tf

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(train_x_selected.shape[1],)),

    tf.keras.layers.Dense(145,activation=tf.nn.relu),

    tf.keras.layers.Dense(100,activation=tf.nn.relu),

    tf.keras.layers.Dense(50,activation=tf.nn.relu),

    tf.keras.layers.Dense(10,activation=tf.nn.relu),



    tf.keras.layers.Dense(1,activation=tf.nn.sigmoid),

])
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model.fit(train_x_selected,train_y_dummy,epochs=300,verbose=True)

test_loss, test_acc = model.evaluate(test_x_selected, test_y_dummy)

test_acc # here we reach a test score of 0.78 which is slightly better than others.
df_feature
df_feature['Survival_Rate'] =  0 # set default value



for idx,row in df_feature.iterrows():

    if (df_feature.loc[idx,'Sex_Coded_1'] == 0): # if female 

        df_feature.loc[idx,'Survival_Rate'] = 1

    

    if(df_feature.loc[idx,'Sex_Coded_1'] == 0) & (df_feature.loc[idx,'Pclass_3'] == 1) & (df_feature.loc[idx,'FareBin_Coded_3']):

        df_feature.loc[idx,'Survival_Rate'] = 0

        

    if(df_feature.loc[idx,'Sex_Coded_1'] == 0) & (df_feature.loc[idx,'Title_Coded_3'] == 0) & (df_feature.loc[idx,'Title_Coded_2'] == 0) & (df_feature.loc[idx,'Title_Coded_1'] == 0) & (df_feature.loc[idx,'Title_Coded_4'] == 0):

        df_feature.loc[idx,'Survival_Rate'] = 1

poly = PolynomialFeatures(degree=2).fit(df_feature)

df_transformed = poly.transform(df_feature)
len(poly.get_feature_names())
highest_score= 0

model = XGBClassifier(n_estimators= 10, max_depth= 6, learning_rate = 0.1)

for i in range(1,df_transformed.shape[1] + 1,1):

    select = SelectKBest(score_func=chi2,k=i)

    select.fit(df_transformed,df1[y])

    df_selected = select.transform(df_transformed)

    

    model.fit(df_selected,df1[y])

    scores = cross_val_score(model,df_selected,df1[y],cv=cv_split) # using xgboost model

    print('features: %i, score %.3f +/- %.3f' % (i,np.mean(scores),np.std(scores)))

    

    

    # save feature with highest score

    

    if np.mean(scores) > highest_score:

        highest_score = np.mean(scores)

        std = np.std(scores)

        number_features = i

    elif np.mean(scores) == highest_score:

        if np.std(scores) < std:

            highest_score = np.mean(scores)

            std = np.std(scores)

            number_features = i

print('best number of features  %i,  with score: %.3f +/- %.3f' %(number_features,highest_score,std))
select = SelectKBest(score_func=chi2, k=number_features)

select.fit(df_transformed, df1[y])

df_selected = select.transform(df_transformed)
# # important modules needed for GP

# import pickle

# import operator

# import math

# import string

# import random



# from deap import algorithms, base, creator, tools, gp

# nb_iterations = 1000 # Warning Can be very computationally expensive. Set to high values  only if you can wait.





# def protectedDiv(left, right):

#     '''

#     This is a function that replaces normal division to prevent zero division errors 

#     '''

#     try:

#         return left / right

#     except ZeroDivisionError:

#             return left





#     # next step is selecting primitives (operators such as addition, multiplication etc)



# pset = gp.PrimitiveSet('Main', df_selected.shape[1])

# pset.addPrimitive(operator.add,2) #  addprimitive takes 2 arguments: the operator and number of operations (e.g you add 2 numbers)

# pset.addPrimitive(operator.sub, 2)

# pset.addPrimitive(operator.mul, 2)

# pset.addPrimitive(protectedDiv,2) # note we use protected div instead of normal div

# pset.addPrimitive(math.cos, 1)

# pset.addPrimitive(math.sin, 1)

# pset.addPrimitive(max, 2)

# pset.addPrimitive(min, 2)

# pset.addPrimitive(math.tanh,1)

# pset.addPrimitive(abs,1)







# pset.renameArguments(ARG0='x1')

# pset.renameArguments(ARG1='x2')

# pset.renameArguments(ARG2='x3')

# pset.renameArguments(ARG3='x4')

# pset.renameArguments(ARG4='x5')

# pset.renameArguments(ARG5='x6')

# pset.renameArguments(ARG6='x7')

# pset.renameArguments(ARG7='x8')

# pset.renameArguments(ARG8='x9')

# pset.renameArguments(ARG9='x10')

# pset.renameArguments(ARG10='x11')

# pset.renameArguments(ARG11='x12')

# pset.renameArguments(ARG12='x13')

# pset.renameArguments(ARG13='x14')

# pset.renameArguments(ARG14='x15')

# pset.renameArguments(ARG15='x16')

# pset.renameArguments(ARG16='x17')

# pset.renameArguments(ARG17='x18')

# pset.renameArguments(ARG18='x19')

# pset.renameArguments(ARG19='x20')

# pset.renameArguments(ARG20='x21')

# pset.renameArguments(ARG21='x22')

# pset.renameArguments(ARG22='x23')

# pset.renameArguments(ARG23='x24')

# pset.renameArguments(ARG24='x25')

# pset.renameArguments(ARG25='x26')

# pset.renameArguments(ARG26='x27')

# pset.renameArguments(ARG27='x28')

# pset.renameArguments(ARG28='x29')

# pset.renameArguments(ARG29='x30')

# pset.renameArguments(ARG30='x31')

# pset.renameArguments(ARG31='x32')

# pset.renameArguments(ARG32='x33')

# pset.renameArguments(ARG33='x34')

# pset.renameArguments(ARG34='x35')

# pset.renameArguments(ARG35='x36')

# pset.renameArguments(ARG36='x37')

# pset.renameArguments(ARG37='x38')

# pset.renameArguments(ARG38='x39')

# pset.renameArguments(ARG39='x40')

# pset.renameArguments(ARG40='x41')

# pset.renameArguments(ARG41='x42')

# pset.renameArguments(ARG42='x43')

# pset.renameArguments(ARG43='x44')

# pset.renameArguments(ARG44='x45')

# pset.renameArguments(ARG45='x46')

# pset.renameArguments(ARG46='x47')

# pset.renameArguments(ARG47='x48')

# pset.renameArguments(ARG48='x49')

# pset.renameArguments(ARG49='x50')



# def randomString(stringLength=10):

#     """Generate a random string of fixed length """

#     letters = string.ascii_lowercase

#     return ''.join(random.choice(letters) for i in range(stringLength))



# pset.addEphemeralConstant(randomString(), lambda: random.uniform(-10,10)) # ephermal constant used to terminate the terminal





# # create fitness and genotype 



# creator.create("FitnessMin", base.Fitness, weights=(1.0,))

# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)



# # we want to register some parameters specific to the evolution process,using the toolbox

# # some code taken from https://www.kaggle.com/guesejustin/91-genetic-algorithms-explained-using-geap



# toolbox = base.Toolbox()

# toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("compile", gp.compile, pset=pset)



# inputs =  df_selected.tolist() #df1[x_rfe].values.tolist()

# outputs = df1[y].values.tolist()

# length_input = len(inputs) +1 



# def evalSymbReg(individual):

# # Transform the tree expression in a callable function

#     func = toolbox.compile(expr=individual)

#     # Evaluate the accuracy of individuals // 1|0 == survived

#     return math.fsum(np.round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs)) / length_input,

# toolbox.register("evaluate", evalSymbReg)

# toolbox.register("select", tools.selTournament, tournsize=3)

# toolbox.register("mate", gp.cxOnePoint)

# toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))



# # stats functions 





# stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

# stats_size = tools.Statistics(len)

# mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

# mstats.register("avg", np.mean)

# mstats.register("std", np.std)

# mstats.register("min", np.min)

# mstats.register("max", np.max)





# pop = toolbox.population(n=300)

# hof = tools.HallOfFame(1)



# pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=nb_iterations, stats=mstats,

#                                halloffame=hof, verbose=True)

    

# result = toolbox.compile(expr=hof[0]) 

# result
import tensorflow as tf

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(df_selected.shape[1],)),

    tf.keras.layers.Dense(200,activation=tf.nn.relu),

    tf.keras.layers.Dense(100,activation=tf.nn.relu),

    tf.keras.layers.Dense(50,activation=tf.nn.relu),

    tf.keras.layers.Dense(10,activation=tf.nn.relu),



    tf.keras.layers.Dense(1,activation=tf.nn.sigmoid),

])

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

history = model.fit(df_selected,df1[y],epochs=30,verbose=True)

test_loss, test_acc = model.evaluate(df_selected,df1[y])

test_acc
df2_feature = pd.get_dummies(df2[x_rfe].astype(str),drop_first=1)

df2_feature['Survival_Rate'] =  0 # set default value



for idx,row in df2_feature.iterrows():

    if (df2_feature.loc[idx,'Sex_Coded_1'] == 0): # if female 

        df2_feature.loc[idx,'Survival_Rate'] = 1

    

    if(df2_feature.loc[idx,'Sex_Coded_1'] == 0) & (df2_feature.loc[idx,'Pclass_3'] == 1) & (df2_feature.loc[idx,'FareBin_Coded_3']):

        df2_feature.loc[idx,'Survival_Rate'] = 0

        

    if(df2_feature.loc[idx,'Sex_Coded_1'] == 0) & (df2_feature.loc[idx,'Title_Coded_3'] == 0) & (df2_feature.loc[idx,'Title_Coded_2'] == 0) & (df2_feature.loc[idx,'Title_Coded_1'] == 0) & (df2_feature.loc[idx,'Title_Coded_4'] == 0):

        df2_feature.loc[idx,'Survival_Rate'] = 1

# poly = PolynomialFeatures(degree=2).fit(df2_feature)

df2_transformed = poly.transform(df2_feature)

df2_selected = select.transform(df2_transformed)

df_test['Survived'] = model.predict_classes(df2_selected)

submit = df_test[['PassengerId','Survived']]



submit.to_csv('../working/submit.csv',index=False)
