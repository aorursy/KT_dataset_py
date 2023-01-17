import pandas as pd
import numpy as np
import scipy as sp
from IPython import display

# common model algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# common model helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# visiualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

import warnings
warnings.filterwarnings('ignore')

import time

%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
# import training and test set
data_raw = pd.read_csv('../input/train.csv')
data_val = pd.read_csv('../input/test.csv')

# Make a copy of the training set to wrangle
data1 = data_raw.copy(deep=True)

# Put both in a list so we can clean both datasets at once
data_cleaner = [data1, data_val]

# Get an overview of the data
print(data_raw.info())
data1.head()
# Check for null values in training and test set
print('Training set null values\n', data1.isnull().sum())
print('-'*20)
print('Test set null values\n', data_val.isnull().sum())
# Check for outliers / weird values
data_raw.describe(include='all')
# Complete data for both training and test sets
for dataset in data_cleaner:
    dataset['Age'].fillna(value=dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(value=dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(value=dataset['Fare'].median(), inplace=True)

# drop columns in test set
drop_columns = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_columns, axis=1, inplace=True)

# check if data completed for both training and test set
print(data1.isnull().sum())
print('-'*20)
print(data_val.isnull().sum())
# Create new features in both the training and test set

for dataset in data_cleaner:
    # Create feature for FamilySize
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
    
    # Create feature for IsAlone. 1 for alone, 0 if there have siblings or children
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    
    # Create Title feature
    dataset['Title'] = dataset['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    
    # Create Age Bins
    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)
    
    # Create Fare Bins
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
# Replace wrong designations
data1['Title'] = data1['Title'].replace({'Ms': 'Miss', 'Mme': 'Mrs', 'Mlle':'Miss'})

# Group uncommon designations as misc
stat_min = 10
title_names = data1['Title'].value_counts() < stat_min

data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
data1.head()
# code categorical data
label = LabelEncoder()

for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    
# define y variable for target/outcome
Target = ['Survived']

# define x variable for original features aka feature selection
data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']

# pretty name/values for charts
data1_x_calc = ['Sex', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age', 'Fare'] # coded for algo calc

data1_xy = Target + data1_x
print('Original X Y: ',data1_xy,'\n' )


# define x variables for original with bin features. Categorical coded numerically instead of in words.
data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')


# Establish dummy variables
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')

data1_dummy.head()
data_val.info()
# Split data into training and test set
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state=0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target], random_state=0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state=0)

print('Data1 Shape:', data1.shape)
print('Train1 Shape:', train1_x.shape)
print('Test1 Shape:', test1_x.shape)

train1_x_bin.head()
# Discrete variable correlation by survival using groupby pivot table
for x in data1_x:
    if data1[x].dtype != 'float64':
        print('Survival Correlation by: ', x)
        print(data1[[x, Target[0]]].groupby(by=x, as_index=False).mean())
        print('-'*40,'\n')
        
# Using crosstabs
print(pd.crosstab(data1['Title'], data1[Target[0]]))
plt.figure(figsize=(16,12))
0
plt.subplot(231)
plt.boxplot(data1['Fare'], meanline=True, showmeans=True);
plt.xlabel('Fare Boxplot')
plt.ylabel('Fare ($)')
plt.title('Fare Boxplot')

plt.subplot(232)
plt.boxplot(data1['Age'], meanline=True, showmeans=True);
plt.xlabel('Age Boxplot')
plt.ylabel('Age (Years)')
plt.title('Age Boxplot')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], meanline=True, showmeans=True);
plt.xlabel('FamilySize Boxplot')
plt.ylabel('Family Size (#)')
plt.title('Family Boxplot')

plt.subplot(234)
plt.hist(x=[data1[data1['Survived'] == 1]['Fare'], data1[data1['Survived'] == 0]['Fare']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.title('Fare Histogram by Survival')
plt.legend()

plt.subplot(235)
plt.hist(x=[data1[data1['Survived'] == 1]['Age'], data1[data1['Survived'] == 0]['Age']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
plt.xlabel('Age (# of years)')
plt.ylabel('# of Passengers')
plt.title('Age Histogram by Survival')
plt.legend()

plt.subplot(236)
plt.hist(x=[data1[data1['Survived'] == 1]['FamilySize'], data1[data1['Survived'] == 0]['FamilySize']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
plt.xlabel('Family Size')
plt.ylabel('# of Passengers')
plt.title('Family Size Histogram by Survival')
plt.legend();
fig, saxis = plt.subplots(2,3,figsize=(16,12))

sns.barplot(x= 'Embarked', y= 'Survived', data=data1, ax= saxis[0,0])
sns.barplot(x = 'Pclass', y='Survived', data=data1, ax=saxis[0,1])
sns.barplot(x = 'IsAlone', y='Survived', data=data1, ax=saxis[0,2], order=[1,0])

sns.pointplot(x='FareBin', y='Survived', data=data1, ax=saxis[1,0])
sns.pointplot(x='AgeBin', y='Survived', data=data1, ax=saxis[1,1])
sns.pointplot(x='FamilySize', y='Survived', data=data1, ax=saxis[1,2])
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(14,12))

sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data1, ax=ax1)
ax1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data1, ax=ax2, split=True)
ax2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x='Pclass', y='FamilySize', hue='Survived', data=data1, ax=ax3)
ax3.set_title('Pclass vs Family Size Survival Comparison')
# graph distribution of qualitative data: sex

fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
qaxis[0].set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax = qaxis[1])
qaxis[1].set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax = qaxis[2])
qaxis[2].set_title('Sex vs IsAlone Survival Comparison');
# More side by side comparisons of FamilySize and Pclass vs the dependent variable

fig, (maxis1, maxis2) = plt.subplots(1,2,figsize=(14,12))

# family size with survival and sex compare
sns.pointplot(x='FamilySize', y='Survived', hue='Sex', data=data1, ax=maxis1, palette={'male': 'blue', 'female': 'pink'},
             markers=['*', 'o'], linestyles=['-', '--'])

# Pclass with survival and sex compare
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data1, ax=maxis2, palette={'male': 'blue', 'female': 'pink'},
             markers=['*', 'o'], linestyles=['-', '--']);
# Plot distribution of passengers who survived vs died
a = sns.FacetGrid(data1, hue='Survived', aspect=4)
a.map(sns.kdeplot, 'Age', shade=True)
a.set(xlim=(0, data1['Age'].max()))
a.add_legend()
# Comparison of Sex, Class, and Age
s = sns.FacetGrid(data1, row='Sex', col='Pclass', hue='Survived')
s.map(plt.hist, 'Age', alpha=0.5)
s.add_legend()
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

correlation_heatmap(data1)
# Machine Learning Algo Selection and Initialization
MLA = [
    # Ensemble methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    # Gaussian Process
    gaussian_process.GaussianProcessClassifier(),
    
    #General Linear Models
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    # Naive Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    # Nearest Neighbour
    neighbors.KNeighborsClassifier(),
    
    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    # XGBoost
    XGBClassifier()
]

# Split dataset in cross-validation with sklearn shuffle split. This is an alternative to train_test_split
# run the model 10x, with a train/test split of 60/30, intentionally leaving out 10%
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

# Create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

# Create table to compare MLA predictions
MLA_predict = data1[Target]

# loop through MLA and save performance in the table
row_index = 0
for alg in MLA:
    
    # Set names and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    # Score model with cross validation
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    
    # Find std, and if this is non-bias random sample, +/- 3 stdev from mean should statistically
    # capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    
    # Fit and predict for each MLA and save in df
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
    
    row_index += 1
    
# Print and sort table
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
MLA_compare
# Plot results of each classifier
plt.figure(figsize=(16,12))
sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='m')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm');
# Intialise base model
dtree = tree.DecisionTreeClassifier(random_state=0)
base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv=cv_split)
dtree.fit(data1[data1_x_bin], data1[Target])

print('Before DT Parameters', dtree.get_params())
print('Before DT Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100) )
print('Before DT Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('Before DT Test w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std()*100*3))
print('-'*60)

# Tune hyper parameters
# Use Gridsearch to get run all permutations of the hyper parameters. Put permutations in dict first
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth':[2,4,6,8,10,None], 'random_state': [0]}

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=cv_split)
tune_model.fit(data1[data1_x_bin], data1[Target])

print('Afrer DT Parameters', tune_model.best_params_)
print('After DT Training w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('After DT Test w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('After DT Test w/bin score 3*std: +/- {:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
# Get data from base model for comparison
print('Before DT RFE Training Shape Old', data1[data1_x_bin].shape)
print('Before DT RFT Training Columns Old', data1[data1_x_bin].columns.values)

print('Before DT RFE Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100) )
print('Before DT RFE Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('Before DT RFE Test w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std()*100*3))
print('-'*60)

# Feature selection
dtree_rfe = feature_selection.RFECV(dtree, step=1, cv=cv_split, scoring='accuracy')
dtree_rfe.fit(data1[data1_x_bin], data1[Target])

# get predictors after RFE
X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]

# Use the columns, and run the dtree again scoring using CV
rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target], cv=cv_split)

# Print out results
print('After DT RFE Training Shape New: ', data1[X_rfe].shape)
print('After DT RFE Training Columns New: ', X_rfe)

print('After DT RFE Training w/bin score mean: {:.2f}'.format(rfe_results['train_score'].mean()*100))
print('After DT RFE Test w/bin score mean: {:.2f}'.format(rfe_results['test_score'].mean()*100))
print('After DT RFE Test w/bin score 3*std: +/- {:.2f}'.format(rfe_results['test_score'].std()*100*3))
print('-'*60)

# Tune RFE model with gridsearch
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split)
rfe_tune_model.fit(data1[X_rfe], data1[Target])

# Print out results of tuned DT RFE model
print('After DT RFE Parameters', rfe_tune_model.best_params_)
print('After RFE DT Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][rfe_tune_model.best_index_]*100))
print('After RFE DT Test w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][rfe_tune_model.best_index_]*100))
print('After RFE DT Test w/bin score 3*std: +/- {:.2f}'.format(rfe_tune_model.cv_results_['std_test_score'][rfe_tune_model.best_index_]*100*3))
correlation_heatmap(MLA_predict)
# the blue and light red models have low corr, can create an ensemble of models 
# Create the list of models
vote_est = [
    
    # Ensemble methods
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    
    # Gaussian Processes
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    # General Linear Model
    ('lr', linear_model.LogisticRegressionCV()),
    
    # Naive Bayes
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    # Nearest Neighbour
    ('knn', neighbors.KNeighborsClassifier()),
    
    # SVM
    ('svc', svm.SVC(probability=True)),
    
    # XGBoost
    ('xgb', XGBClassifier())
]

# Create a hard vote ensemble (majority rules)
vote_hard = ensemble.VotingClassifier(vote_est, voting='hard', n_jobs=-1)
vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], cv=cv_split)
vote_hard.fit(data1[data1_x_bin], data1[Target])

# Print out results for vote hard classifier
print('Hard voting training score mean: {:.2f}'.format(vote_hard_cv['train_score'].mean()*100))
print('Hard voting test score mean: {:.2f}'.format(vote_hard_cv['test_score'].mean()*100))
print('Hard voting 3*std: +/- {:.2f}'.format(vote_hard_cv['test_score'].std()*100*3))
print('-'* 60)

# Create a soft vote ensemble (weighted probabilities)
vote_soft = ensemble.VotingClassifier(vote_est, voting='soft', n_jobs=-1)
vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv= cv_split)
vote_soft.fit(data1[data1_x_bin], data1[Target])

# Print out results for soft voting classifer
print('Soft voting training score mean: {:.2f}'.format(vote_soft_cv['train_score'].mean()*100))
print('Soft voting test score mean: {:.2f}'.format(vote_soft_cv['test_score'].mean()*100))
print('Soft voting 3*std: +/- {:.2f}'.format(vote_soft_cv['test_score'].std()*100*3))
# Warning, running is computationally expensive

grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
    # Adaboost Classifier
    [{
        'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
    }],
    
    # Bagging Classifier
    [{'n_estimators': grid_n_estimator,
      'max_samples': grid_ratio,
      'random_state': grid_seed
    }],
    
    # ExtraTreesClassifier
    [{'n_estimators': grid_n_estimator,
      'criterion': grid_criterion,
      'max_depth': grid_max_depth,
      'random_state': grid_seed
    }],
    
    # GradientBoostingClassifier
    [{'learning_rate': grid_learn,
      'n_estimators': grid_n_estimator,
      'max_depth': grid_max_depth,
      'random_state': grid_seed
    }],
    
    # RandomForestClassifier
    [{'n_estimators': grid_n_estimator,
      'criterion': grid_criterion,
      'max_depth': grid_max_depth,
      'oob_score': [True],
      'random_state': grid_seed
    }],
    
    # GaussianProcessClassifier
    [{'max_iter_predict': grid_n_estimator,
    'random_state': grid_seed
     }],
    
    # Logistic Regression
    [{'fit_intercept': grid_bool,
      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
      'random_state': grid_seed
    }],
    
    # BernoulliNB
    [{'alpha': grid_ratio
    }],
    
    # GaussianNB
    [{}],
    
    # K Nearest Neighbours
    [{'n_neighbors': [1,2,3,4,5,6,7],
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }],
    
    # SVM
    [{'C': [1,2,3,4,5],
      'gamma': grid_ratio,
      'decision_function_shape': ['ovo', 'ovr'],
      'probability': [True],
      'random_state': grid_seed
    }],
    
    #XGBoost
    [{'learning_rate': grid_learn,
      'max_depth': [1,2,4,6,8,10],
      'n_estimators': grid_n_estimator,
      'seed': grid_seed
    }]
]


start_total = time.perf_counter()
for clf, param in zip(vote_est, grid_param):
    
    start = time.perf_counter()
    best_search = model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc')
    best_search.fit(data1[data1_x_bin], data1[Target])
    run = time.perf_counter() - start
    
    best_param = best_search.best_params_
    print('The best parameter for {} is {}, runtime: {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param)
    
run_total = time.perf_counter() - start_total
print('Total optimization time is {:.2f} minutes'.format(run_total/60))
print('-'*60)
# Create a hard vote ensemble (majority rules)
grid_hard = ensemble.VotingClassifier(vote_est, voting='hard', n_jobs=-1)
grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target], cv=cv_split)
grid_hard.fit(data1[data1_x_bin], data1[Target])

# Print out results for vote hard classifier
print('Hard voting (tuned hyperparameters) training score mean: {:.2f}'.format(grid_hard_cv['train_score'].mean()*100))
print('Hard voting (tuned hyperparameters) test score mean: {:.2f}'.format(grid_hard_cv['test_score'].mean()*100))
print('Hard voting (tuned hyperparameters) 3*std: +/- {:.2f}'.format(grid_hard_cv['test_score'].std()*100*3))
print('-'* 60)

# Create a soft vote ensemble (weighted probabilities)
grid_soft = ensemble.VotingClassifier(vote_est, voting='soft', n_jobs=-1)
grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target], cv= cv_split)
grid_soft.fit(data1[data1_x_bin], data1[Target])

# Print out results for soft voting classifer
print('Soft voting (tuned hyperparameters) training score mean: {:.2f}'.format(grid_soft_cv['train_score'].mean()*100))
print('Soft voting (tuned hyperparameters) test score mean: {:.2f}'.format(grid_soft_cv['test_score'].mean()*100))
print('Soft voting (tuned hyperparameters) 3*std: +/- {:.2f}'.format(grid_soft_cv['test_score'].std()*100*3))