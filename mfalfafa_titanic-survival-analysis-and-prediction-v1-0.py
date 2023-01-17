# Load libraries or packages for Machine Learning purposes

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



#misc libraries

import random

import time





#ignore warnings

import warnings

warnings.filterwarnings('ignore')



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

import lightgbm as lgb



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.impute import SimpleImputer



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
# Step 3. Data preprocessing

# load data

train_data_full=pd.read_csv('../input/titanic/train.csv')

test_data_full=pd.read_csv('../input/titanic/test.csv')



# omit missing values on target data (Survived)

train_data_full.dropna(subset=['Survived'], axis=0, inplace=True)



# preview data

# print(train_data_full.head())

# print(test_data_full.head())



# view missing values and summary of data

train_miss_cols=train_data_full.columns[train_data_full.isnull().sum()>0]

test_miss_cols=test_data_full.columns[test_data_full.isnull().sum()>0]

print('-'*10)

print('Missing values on train data:')

print(train_data_full[train_miss_cols].isnull().sum())

print('-'*10)

print('Train data summary:')

print(train_data_full.describe())

print('-'*10)

print('Missing values on test data:')

print(test_data_full[test_miss_cols].isnull().sum())

print('-'*10)

print('Test data summary:')

print(test_data_full.describe())



# ignore/drop Cabin column because there are so many missing values

# drop PassengerId in train data because it's just an ID for testing

# drop Ticket because it's determined by Pclass

train_data_full.drop(['Cabin','PassengerId','Ticket'], axis=1, inplace=True)

test_data_full.drop(['Cabin','Ticket'], axis=1, inplace=True)

train_miss_cols=train_miss_cols.drop(['Cabin'])

test_miss_cols=test_miss_cols.drop(['Cabin'])



# data cleaner for data preprocessing

data_cleaner=[train_data_full, test_data_full]



# show distribution on data that have missing values to decide the type of categorical encoder that fits to data that have missing values

# filter categorical columns that have missing values 

train_miss_cat=train_data_full[[i for i in train_miss_cols if train_data_full[i].dtypes=='object']]

test_miss_cat=test_data_full[[i for i in test_miss_cols if test_data_full[i].dtypes=='object']]

miss_cat_data=[train_miss_cat, test_miss_cat]



# encode with label encoder for categorical columns in missing columns 

def label_encoder(df):

    label_encoder = LabelEncoder()

    for col_name in df.columns:

        series = df[col_name]

        df[col_name] = pd.Series(

            label_encoder.fit_transform(series[series.notnull()]),

            index=series[series.notnull()].index

        )

    return df

encoded_miss_cat_data=[label_encoder(i) for i in miss_cat_data]

# join numeric type with encoded categorical columns in train/test data that have missing values

train_miss=train_data_full[list(set(train_miss_cols)-set(train_miss_cat))].join(encoded_miss_cat_data[0])

test_miss=test_data_full[list(set(test_miss_cols)-set(test_miss_cat))].join(encoded_miss_cat_data[1])



# view data distribution with pairplot seaborn on variables that contain missing values

data_miss=train_data_full[train_miss_cols].join(test_data_full[test_miss_cols].add_suffix('_Test'))

data_miss_plot=train_miss.join(test_miss.add_suffix('_Test'))

skew_val=data_miss_plot.skew(axis=0, skipna=True)

print('Missing values data skewness: \n',skew_val)

g0=sns.pairplot(data_miss_plot, diag_kind='kde')

g0=g0.fig.suptitle('Train and Test data distribution', y=1)



# show boxplot to see skewness and outliers

plt.figure(figsize=(10,5))

plt.subplots_adjust(wspace=0.9)

max_col=len(data_miss_plot.columns)

n=max_col-1

for idx,col in enumerate(data_miss_plot.columns):

    plt.subplot(1,max_col,max_col-n)

    sns.boxplot(data=data_miss_plot[col], showmeans = True, meanline = True)

    plt.title(col)

    n-=1
# impute missing values

# from data distribution and box plot above can be known that:

# Age and Age_Test=positive skewness -> strategy=median

# Embarked=categorical variable -> strategy=mode

# Fare=positive skewness -> strategy=median

imp_med=SimpleImputer(strategy='median')

imp_mod=SimpleImputer(strategy='most_frequent')

for col in skew_val.index.drop('Embarked'):

    data_miss[col]=pd.DataFrame(imp_med.fit_transform(data_miss[[col]]))

data_miss['Embarked']=pd.DataFrame(imp_mod.fit_transform(data_miss[['Embarked']]))

# check for missing values

print(data_miss.isnull().sum())



# join imputed missing values to train/test data

train_miss_imputed=data_miss[train_miss.columns]

test_miss_imputed_cols=list(set(data_miss.columns)-set(train_miss.columns))

test_miss_imputed_cols.sort(reverse=True)

test_miss_imputed=data_miss[test_miss_imputed_cols].rename(columns=dict(zip(test_miss_imputed_cols, test_miss.columns)))

train_data_full.drop(train_miss_imputed.columns, axis=1, inplace=True)

test_data_full.drop(test_miss_imputed.columns, axis=1, inplace=True)

train_data_full=train_data_full.join(train_miss_imputed)

test_data_full=test_data_full.join(test_miss_imputed)

# preview encoded data (train, test) without missing values

print('Check missing values on baseline data')

print('-'*10)

print(train_data_full.isnull().sum())

print('-'*10)

print(test_data_full.isnull().sum())

train_data_full.head()
data_df=[train_data_full, test_data_full]

# cleaning name and extracting Title

for df in data_df:

    df['Title']=df['Name'].str.extract('([A-Za-z]+)\.', expand=True)



# replacing rare Title with more common ones and also drop column Name

mapping={'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

         'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

for df in data_df:

    df.replace({'Title':mapping}, inplace=True)

# preview train dataset

# data_df[0].sample(5)
# add Family_Size in each dataset

for df in data_df:

    df['Family_Size']=df['Parch']+df['SibSp']

# preview train dataset

# data_df[0].sample(5)
# making bins

label=LabelEncoder()

for df in data_df:

    # create bin

    df['FareBin']=pd.qcut(df['Fare'], 5)

    # use label encoder for categorical variables

    df['FareBin_Code']=label.fit_transform(df['FareBin'])

    # drop Fare column

# preview train dataset

# data_df[0].sample(5)
label=LabelEncoder()

for df in data_df:

    df['AgeBin']=pd.qcut(df['Age'], 4)

    df['AgeBin_Code']=label.fit_transform(df['AgeBin'])

# preview train dataset

# data_df[0].sample(5)
# check categorical variables

cat_cols=[]

for df in data_df:

    cat_cols.append([col for col in df.columns if df[col].dtypes=='object'])

# print(cat_cols)

# print('-'*10)

    

# use label encoder to encode categorical variables

label=LabelEncoder()

train_encoded=data_df[0][cat_cols[0]].apply(label.fit_transform)

test_encoded=data_df[1][cat_cols[1]].apply(label.fit_transform)



# use one-hot encoder

oh_cols=['Title', 'Sex', 'Embarked']

train_encoded_oh=pd.get_dummies(data_df[0][['Title','Sex','Embarked']]).add_suffix('_OH')

test_encoded_oh=pd.get_dummies(data_df[1][['Title','Sex','Embarked']]).add_suffix('_OH')

OH_encoded_cols=train_encoded_oh.columns



# join encoded columns to dataset

train_data_full=data_df[0].join(train_encoded.add_suffix('_Code'))

test_data_full=data_df[1].join(test_encoded.add_suffix('_Code'))

# join one-hot encoded columns to dataset

train_data_full=train_data_full.join(train_encoded_oh)

test_data_full=test_data_full.join(test_encoded_oh)



# select features for baseline and engineered dataset

baseline_feature_cols=['Pclass', 'Sex_Code', 'Embarked_Code', 'Title_Code', 'Family_Size']

engineered_feature_cols=['Pclass', 'Family_Size', 'FareBin_Code', 'AgeBin_Code']+list(OH_encoded_cols)

train_data_full[engineered_feature_cols].head()
# create function for data splits and train model

def get_data_splits(df, valid_fraction=0.1):

    valid_size=int(len(df)*valid_fraction)

    train=df[:-valid_size*2]

    valid=df[-valid_size*2:-valid_size]

    test=df[-valid_size:]

    return train,valid,test



# split training data for train, valid and test

train_,valid_,test_=get_data_splits(train_data_full)
# discrete variable correlation on Train data by survival using groupby

for col in train_data_full[['Pclass','Sex','Embarked', 'FareBin', 'AgeBin', 'Title']]:

    print('Survival correlation by: ', col)

    print(train_data_full[[col, 'Survived']].groupby(col, as_index=False).mean())

    print('-'*10)
# graph distribution for quantitative data (Age, Fare, Family size) on training data

plt.figure(figsize=(14,12))

# plt.subplots_adjust(wspace=0.9, hspace=0.3)



# Fare boxplot

plt.subplot(231)

sns.boxplot(data=train_data_full['Fare'], showmeans=True, meanline=True)

plt.title('Fare Distribution')

plt.ylabel('Fare ($)')

# Age subplot

plt.subplot(232)

sns.boxplot(data=train_data_full['Age'], showmeans=True, meanline=True)

plt.title('Age Distribution')

plt.ylabel('Age (Years)')

# Family size boxplot

plt.subplot(233)

sns.boxplot(data=train_data_full['Family_Size'], showmeans=True, meanline=True)

plt.title('Family size Distribution')

plt.ylabel('Family size (#))')

# Fare vs survived histogram plot

plt.subplot(234)

plt.hist(x=[train_data_full[train_data_full['Survived']==1]['Fare'], 

           train_data_full[train_data_full['Survived']==0]['Fare']], color=['g','r'], label=['Survived', 'Dead'], bins=20)

plt.title('Fare by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of passengers')

plt.legend()

# Age vs survival

plt.subplot(235)

plt.hist(x=[train_data_full[train_data_full['Survived']==1]['Age'], 

           train_data_full[train_data_full['Survived']==0]['Age']], color=['g','r'], label=['Survived', 'Dead'], bins=30)

plt.title('Age by Survival')

plt.xlabel('Age (years)')

plt.ylabel('# of passengers')

plt.legend()

# Family size vs survival

plt.subplot(236)

plt.hist(x=[train_data_full[train_data_full['Survived']==1]['Family_Size'], 

           train_data_full[train_data_full['Survived']==0]['Family_Size']], color=['g','r'], label=['Survived', 'Dead'], bins=20)

plt.title('Family size by Survival')

plt.xlabel('Family size (#)')

plt.ylabel('# of passengers')

_=plt.legend()
# graph for categorical data (Title, Sex, Pclass, Embarked, FareBin, AgeBin)

# graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(14,12))

sns.barplot(x='Title', y='Survived', data=train_data_full, ax=saxis[0,0])

saxis[0,0].set_title('Title vs Survived')

sns.barplot(x='Sex', y='Survived', data=train_data_full, ax=saxis[0,1])

saxis[0,1].set_title('Sex vs Survived')

sns.barplot(x='Pclass', y='Survived', data=train_data_full, ax=saxis[0,2])

saxis[0,2].set_title('Pclass vs Survived')

sns.barplot(x='Embarked', y='Survived', data=train_data_full, ax=saxis[1,0])

saxis[1,0].set_title('Title vs Embarked')

farebinplot=sns.barplot(x='FareBin', y='Survived', data=train_data_full, ax=saxis[1,1])

farebinplot.set_xticklabels(farebinplot.get_xticklabels(), rotation=45, horizontalalignment='right')

saxis[1,1].set_title('FareBin vs Survived')

sns.barplot(x='AgeBin', y='Survived', data=train_data_full, ax=saxis[1,2])

_=saxis[1,2].set_title('AgeBin vs Survived')
# graph distribution of qualitative data Pclass compared to other features

# Pclass is mattered for survival

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,7))



sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=train_data_full, ax=axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')

sns.boxplot(x='Pclass', y='Age', hue='Survived', data=train_data_full, ax=axis2)

axis2.set_title('Pclass vs Age Survival')

sns.boxplot(x='Pclass', y='Family_Size', hue='Survived', data=train_data_full, ax=axis3)

_=axis3.set_title('Pclass vs Family Size Survival Comparison')
# graph distribution of qualitative data Sex compared to other features

# Sex is mattered for survival

fig, saxis = plt.subplots(1,3,figsize=(14,7))



sns.barplot(x='Sex', y='Survived', hue='Embarked', data=train_data_full, ax=saxis[0])

saxis[0].set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train_data_full, ax=saxis[1])

saxis[1].set_title('Sex vs Pclass Survival')

sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train_data_full, ax=saxis[2])

_=saxis[2].set_title('Sex vs Pclass Survival')
# pairplot of entire dataset

display_cols=['Survived','Pclass','SibSp','Parch','Fare', 'Age','Family_Size','FareBin_Code', 'AgeBin_Code', 'Title_Code']

pp=sns.pairplot(train_data_full[display_cols], hue='Survived', palette='deep', size=1.2, diag_kind='kde',diag_kws=dict(shade=True), plot_kws=dict(s=10))

for axis in pp.fig.axes:   # get all the axis

    axis.set_xlabel(axis.get_xlabel(), rotation=45)

_=pp.set(xticklabels=[])
# heatmap correlation of train dataset

def heatmap_correlation(df):

    _, ax=plt.subplots(figsize=(14,12))

    colormap=sns.diverging_palette(220, 10, as_cmap=True)

    _=sns.heatmap(

        df.corr(),

        cmap=colormap,

        square=True,

        cbar_kws={'shrink':.9},

        ax=ax,

        annot=True,

        linewidths=0.1, vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12}

    )

    plt.title('Pearson Correlation of features', y=1.05, size=15)

heatmap_correlation(train_data_full[display_cols])
# Machine Learning Algorithm (MLA) selection and initialization

MLA=[

    # Ensemble methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    # XGBoost

    XGBClassifier(),

    

    # LightGBM

    lgb.LGBMClassifier(),

    

    # Gaussian process

    gaussian_process.GaussianProcessClassifier(),

    

    # GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    # Naive bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    # Nearest neighbors

    neighbors.KNeighborsClassifier(),

    

    # SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    # Trees

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    # Discrimant analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),

]



# split dataset in cross-validation with the splitter class

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0 )



# table to compare MLA metrics

MLA_columns=['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 

            'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']

MLA_compare=pd.DataFrame(columns=MLA_columns)



models={}



# feature cols

# cols=[col for col in baseline_feature_cols if col!='Survived']

cols=[col for col in engineered_feature_cols if col!='Survived']



# iterate through MLA and save performance to table

for idx, alg in enumerate(MLA):

    # set name and params

    MLA_name=alg.__class__.__name__

    MLA_compare.loc[idx, 'MLA Name']=MLA_name

    MLA_compare.loc[idx, 'MLA Parameters']=str(alg.get_params())

    

    # score model with cross validation

    cv_results=model_selection.cross_validate(alg, train_data_full[cols], train_data_full['Survived'], cv=cv_split, return_train_score=True)

    MLA_compare.loc[idx, 'MLA Time']=cv_results['fit_time'].mean()

    MLA_compare.loc[idx, 'MLA Train Accuracy Mean']=cv_results['train_score'].mean()

    MLA_compare.loc[idx, 'MLA Test Accuracy Mean']=cv_results['test_score'].mean()

    MLA_compare.loc[idx, 'MLA Test Accuracy 3*STD']=cv_results['test_score'].std()*3

    

    # save MLA predictions score and model

    models[MLA_name]=[cv_results['test_score'].mean(), alg]



# show and sort table

MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)

MLA_compare

    
# barplot for MLA comparison

ax=sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='#52518f')

test_acc=MLA_compare['MLA Test Accuracy Mean'].apply(lambda x: f'{x*100:.2f}')

target_name=MLA_compare['MLA Name']+' ('+ test_acc +'%)'

ax.set_yticklabels(target_name)



# pretify using pyplot

plt.title('Machine Learning Algorithm Test Accuracy Score\n')

plt.xlabel('Accuracy Score (%)')

_=plt.ylabel('Algorithm')

# hyper-parameters tune with GridSearchCV

grid_ratio=[.1, .25, .5, .75, 1.0]

best_models={}



estimators=[

    ('svc', svm.SVC()),

    ('nusvc', svm.NuSVC()),

    ('rc', linear_model.RidgeClassifierCV()),

    ('lsvc', svm.LinearSVC()),

    ('lda', discriminant_analysis.LinearDiscriminantAnalysis())

]



grid_param=[

    [{

        # SVC

        'C': [1,2,3,4,5],

        'gamma': grid_ratio,

        'decision_function_shape': ['ovo', 'ovr'],

        'probability': [True],

        'random_state': [0]

    }],

    [{

        # NuSVC

        'nu': [0.5, 0.7],

        'gamma': grid_ratio,

        'decision_function_shape': ['ovo', 'ovr'],

        'probability': [True, False],

        'random_state': [0]

        

    }],

    [{

        # RidgeClassifierCV

        'alphas':[(0.1, 0.5, 7.0), (0.1, 0.7, 10.0), (0.1, 1.0, 10.0)],

        'normalize':[True, False],

        'scoring':[None],

        'class_weight': ['balanced', None]

    }],

    [{

        # LinearSVC

        'penalty':['l2'],

        'loss':['hinge', 'squared_hinge'],

        'C': [1,2,3,4,5]

    }],

    [{

        # LinearDiscriminantAnalysis

        'solver':['svd', 'lsqr'],

        'shrinkage':[None]

    }]

]



start_total=time.perf_counter()

for clf, param in zip(estimators, grid_param):

    start=time.perf_counter()

    best_search=model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc', n_jobs=-1)

    best_search.fit(train_data_full[cols], train_data_full['Survived'])

    run=time.perf_counter()-start

    

    best_param=best_search.best_params_

    best_score=best_search.best_score_

    best_models[clf[1].__class__.__name__]=[best_score, best_search, run]

    print('Name: ', clf[1].__class__.__name__)

    print('Best score: ', best_score)

    print('best param: ', best_param)

    print('runtime: ', run)

    print('-'*10)

    clf[1].set_params(**best_param)



run_total=time.perf_counter()-start_total

print('Total optimization time: {:.2f} minutes'.format(run_total/60))

print('Finish')
# features selection

for idx, features in enumerate([baseline_feature_cols, engineered_feature_cols]):

    print('='*35)

    print('Features type: ', ['Baseline Features', 'Engineered Features'][idx])

    print('='*35)

    for model in best_models.items():

        model[1][1].best_estimator_.fit(train_[features], train_['Survived'])

        y_pred=model[1][1].best_estimator_.predict(valid_[features])

        score=metrics.roc_auc_score(valid_['Survived'], y_pred)

        print('Model: ', model[0])

        print(f'Validation score: {score:.4f}')

        print('-'*25)
# generate CSV file for submitting survival predictions

model=best_models['SVC'][1]

model.best_estimator_.fit(train_data_full[engineered_feature_cols], train_data_full['Survived'])

y_pred=model.best_estimator_.predict(test_data_full[engineered_feature_cols])



submit=pd.DataFrame({'PassengerId':test_data_full.PassengerId,

                    'Survived':y_pred})

submit.to_csv('submission.csv', index=False)