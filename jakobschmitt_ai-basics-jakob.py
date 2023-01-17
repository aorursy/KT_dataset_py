# Data Processing 

import numpy as np 

import pandas as pd 



# Data Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



mpl.style.use('ggplot')

sns.set_style('dark')

pylab.rcParams['figure.figsize'] = 12,8



# ML Algorithms

import sklearn 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

import xgboost 

from xgboost import XGBClassifier



# Helpful ML Tools

from sklearn.feature_selection import RFECV

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyClassifier



# Bayesian Hyperparameter Optimization

import hyperopt

from hyperopt import STATUS_OK, hp, tpe, fmin, Trials



# Pretty Printing of DataFrames

import IPython

from IPython import display



# Ignore Warnings

import warnings  

warnings.filterwarnings('ignore') 



# Printing the version names for improving reproducibility 

import sys

print("Python version: {}".format(sys.version))

print("NumPy version: {}".format(np.__version__))

print("pandas version: {}".format(pd.__version__))

print("matplotlib version: {}".format(mpl.__version__))

print("seaborn version: {}".format(sns.__version__))

print("scikit-learn version: {}".format(sklearn.__version__))

print("XGBoost version: {}".format(xgboost.__version__))
# Read train and test data

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')



train_data.sample(20)
# inspect numeric features

train_data.describe() 
# inspect alphanumeric features

train_data.describe(include=["O"]) 
# inspect numeric features

test_data.describe() 
# inspect alphanumeric features

test_data.describe(include=["O"]) 
# Calculate and print the overall survival rate in train_data

survival_rate = train_data["Survived"].sum() / len(train_data.index)

print("Survival rate in train data: {}%".format(round(survival_rate * 100, 1)))
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,4))



# Create barplot showing the general distribution of Pclass

counts = sns.countplot(x='Pclass', data=train_data, ax=axes[0])

counts.set_xlabel('Class')

counts.set_ylabel('Count')

counts.set_title('a')



# Calculate survival rates across classes and show them in lineplot

data_plot = train_data.groupby(['Survived', 'Pclass']).size().reset_index().pivot(columns='Survived', index='Pclass', values=0)

ratios = data_plot.iloc[:,1] / data_plot.sum(axis=1)

point = sns.pointplot(x=['1','2','3'], y=ratios, ax=axes[1])

point.set_xlabel('Class')

point.set_ylabel('Survival %')

point.set_title('b')



# Like above, but split by gender and showed presented in a barplot

bar = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_data, ci=None, ax=axes[2]);

bar.set_xlabel('Class')

bar.set_ylabel('Survival %')

bar.set_title('c')



plt.subplots_adjust(wspace=0.3)

plt.show()
# Calculate survival rates split by gender

survival_rate_men = train_data[train_data['Sex'] == 'male']['Survived'].sum() / train_data[train_data['Sex'] == 'male']['Sex'].count()

survival_rate_women = train_data[train_data['Sex'] == 'female']['Survived'].sum() / train_data[train_data['Sex'] == 'female']['Sex'].count()



print('*** Train data ***')

print("Men's survival rate: \t", round(survival_rate_men*100, 1), '%')

print("Women's survival rate: \t", round(survival_rate_women*100, 1), '%')



# Barplot that shows numbers of males and females, while also visualizing survival rates

data_plot = train_data.groupby(['Survived', 'Sex']).size().reset_index().pivot(columns='Survived', index='Sex', values=0)

data_plot.plot(kind='bar', stacked=True, figsize=(6,5))

plt.ylabel('Count')

plt.xticks(rotation=0)

plt.show()
# Inspired by [K1]

# Kernel Density Estimation of the Age distribution, split between survivors and deceased.

# For each class separately.

facet = sns.FacetGrid(train_data, hue='Survived', row='Pclass', aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0, train_data['Age'].max()))

facet.add_legend()

plt.show()
# Survival rates based on family metrics

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))



# Siblings and spouses

sibsp = sns.pointplot(x="SibSp", y="Survived", data=train_data, ci=None, ax=axes[0]);

sibsp.set_title('a');



# Parents and children

parch = sns.pointplot(x="Parch", y="Survived", data=train_data, ci=None, ax=axes[1]);

parch.set_title('b');



# All combined

family = sns.pointplot(x=train_data['SibSp'] + train_data['Parch'], y=train_data['Survived'], ci=None, ax=axes[2]);

family.set_xlabel('SibSp + Parch')

family.set_title('c');
# Boxplots to show Fare distributions across classes

# extreme outlier of Fare>500 was chopped off for better readability

sns.catplot(kind="box", x="Pclass", y="Fare", hue="Survived", data=train_data[train_data["Fare"] < 500], dodge=True, height=6)

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))



# Survival rates of men and women, separated by Embarked

bar1 = sns.barplot(x="Embarked", y="Survived", data=train_data, hue="Sex", ci=None, ax=axes[0]);

bar1.set_title('a');



# Numbers of passengers of the different classes, separated by Embarked

bar2 = sns.countplot(x='Embarked', hue='Pclass', data=train_data, ax=axes[1]);

bar2.set_title('b');



plt.show()
def impute_embarked(df):

    ''' AP 10: Imputing missing Embarked values with training data mode

    '''

    df['Embarked'] = df['Embarked'].fillna(train_data['Embarked'].dropna().mode()[0]) 

    

def numerical_coding(df):

    ''' AP 4, 13: Numerical coding of Sex and Embarked

    '''

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    df['Embarked'] = df['Embarked'].dropna().map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

def impute_age(df):

    ''' AP 5: Imputing missing Age values based on Class-Sex combinations

    '''

    for i in range(2):

        for j in range(1,4):

            c_median = train_data_copy[(train_data_copy['Sex'] == i) & (train_data_copy['Pclass'] == j)]['Age'].dropna().median()

            df.loc[(df['Age'].isnull()) & (df['Sex'] == i) & (df['Pclass'] == j), 'Age'] = c_median



def extract_title(df):

    ''' AP 3: Extract title information, simplify it, and encode numerically. 

    '''

    df['Title'] = df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    Title_Dictionary = { # taken from [K1]

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"

                    }

    df['Title'] = df['Title'].map(Title_Dictionary)

    df['Title'] = df['Title'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Royalty': 4, 'Officer': 5}).astype(int)

    

def create_English_feature(df):

    '''AP 3b: Is_English_native is 1 if the estimated country of origin is English-speaking, and 0 otherwise

    '''

    # Alpha codes of English-speaking countries [S6][S7]:

    en_countries = ['AG', 'AU', 'BS', 'BB', 'BZ', 'CA', 'DM', 'GD', 

                    'GY', 'IE', 'JM', 'NZ', 'KN', 'TT', 'GB', 'US']

    df['Is_English_native'] = df['Country'].isin(en_countries) * 1



def bin_variables(df):

    ''' AP 6, 10: Prevent overfitting by binning continuous variables

    '''

    # Bin Fare in equally populated bins, based on train data

    df['FareBin'] = pd.cut(x=df['Fare'], bins=qbins_Fare, labels=['0', '1', '2', '3']).astype(int)

    

    # Bin Age according to natural bins

    BINS = (0, 5, 12, 18, 35, 60, 100)

    # bins correspond to categories:

    # Baby, Child, Teenager, Young Adult, Adult, Senior (modified from [K3]) 

    df['AgeBin'] = pd.cut(df['Age'], BINS, labels=['0', '1', '2', '3', '4', '5']).astype(int)



def simplify_family(df):

    ''' AP 8: Process the Parch and SibSp information to create simple and meaningful family size features (inspired by [K1])

    '''

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    

    # For passengers traveling alone

    df['Family_Alone'] = df['FamilySize'].map(lambda n: (n == 1) * 1)

    

    # For passengers traveling with 1 to 3 family members

    df['Family_Small'] = df['FamilySize'].map(lambda n: (2 <= n <= 4) * 1)

    

    # For passengers traveling with more than 3 family members

    df['Family_Large'] = df['FamilySize'].map(lambda n: (5 <= n) * 1)

    

def simplify_fare(df):

    ''' AP 11: Set Fare values to 0 for second and third class to prevent overfitting 

    '''

    df[df['Pclass'] != 1]['FareBin'] = 0

    

def create_interactions(df):

    ''' AP 2, 7: Create Pclass_x_Sex and Pclass_x_AgeBin interactions 

    '''

    df['Pclass_x_Sex'] = df['Pclass'] * df['Sex']

    df['Pclass_x_AgeBin'] = df['Pclass'] * df['AgeBin']



def drop_unnecessary_features(df):

    ''' AP 9, 12: Drop features that are no longer needed

    '''

    to_drop = ['Ticket', 'Cabin', 'Name', 'Parch', 'SibSp', 'Age', 'Fare'] 

    df.drop(to_drop, axis=1, inplace=True)

    

def create_dummies(df):

    ''' AP 2, 6, 13: Create dummies for Title, AgeBin, Embarked 

    '''

    # Make copies so that original data doesn't get lost

    df['Title_copy'] = df['Title']

    df['AgeBin_copy'] = df['AgeBin']

    df['Embarked_copy'] = df['Embarked']

    return pd.get_dummies(df, columns=['Title_copy', 'AgeBin_copy', 'Embarked_copy'], prefix=['Title', 'Age', 'Embarked'], drop_first=True)

    

def process_df(df):

    ''' Summarizes all processing parts

    ''' 

    impute_embarked(df)

    numerical_coding(df)

    impute_age(df)

    extract_title(df)

    create_English_feature(df)

    bin_variables(df)

    simplify_family(df)

    simplify_fare(df)

    create_interactions(df)

    drop_unnecessary_features(df)

    return create_dummies(df)



# AP 1: Drop PassengerId in the training data

train_data.drop(['PassengerId'], axis=1, inplace=True)



# AP 14: Impute missing Fare value in test_kaggle based on training data median of the according Pclass

pclass = test_data.loc[test_data['Fare'].isnull(), 'Pclass'].values[0]

class_median = train_data[train_data['Pclass'] == pclass]['Fare'].median()

test_data.loc[test_data['Fare'].isnull(), 'Fare'] = class_median



# AP 3b: Load country-of-origin data

# Manually acquired from NamSor API [A1]

Country = pd.read_csv('../input/country/country.csv')

train_data['Country'] = Country.iloc[:len(train_data)]

test_data['Country'] = Country.iloc[len(train_data):].values



# To prevent data leakage, some functions must use training data when imputing testing data. 

# Due to the various processing steps, a training data copy is needed.

train_data_copy = train_data.copy()

impute_embarked(train_data_copy)

numerical_coding(train_data_copy)



# Get equally populated bins from first class passengers for binning the Fare feature 

_, qbins_Fare = pd.qcut(train_data_copy[train_data_copy['Pclass'] == 1]['Fare'], 4, retbins=True)

qbins_Fare[0], qbins_Fare[-1] = -1, 1000



# Process all relevat data frames 

train_data = process_df(train_data)

test_data = process_df(test_data)
train_data.sample(5)
# Set scoring metric for:

# 1. Automatic feature selection

# 2. Hyperparameter optimization

# 3. Model evaluation



SCORING_METHOD = 'f1'



# Set random state for all functions with an element of randomness

# For better reproducibility



RANDOM_STATE = 42



# Select the number of folds for n-fold cross validation

# 3 to 10 are reasonable values



CV = 5
# Define short names and function calls

models = {

    'LogReg':        'LogisticRegression', 

    'SVM' :          'SVC', 

    'DecisionTree':  'DecisionTreeClassifier', 

    'RandForest':    'RandomForestClassifier', 

    'XGB':           'XGBClassifier',

    'KNN':           'KNeighborsClassifier'

}



# Create default classifier instances

for name, classifier in models.items():

    exec(name + '=' + classifier +'()')



# Set SVM kernel to 'linear'

SVM.kernel = 'linear'



# Set Random States for models including some element of randomness 

DecisionTree.random_state = RANDOM_STATE

RandForest.random_state = RANDOM_STATE

XGB.random_state = RANDOM_STATE



# Algorithms that can deal with categorical data

cat_models = ['DecisionTree', 'RandForest', 'KNN', 'XGB']



# KNN doesn't provide feature importance metrics. 

# Hence, feature selection needs to be handled separately.
# Define potential sets of features

# Best subsets will later be selected automatically for LogReg and SVM



# For models that can deal with categorical variables with various values

features_cat = ['Sex', 'Title', 'AgeBin', 'Pclass', 'FamilySize', 'FareBin', 'Embarked', 'Is_English_native']



# For models that need dummy coding

features_dummy = ['Sex', 'Family_Alone', 'Family_Small', 'Pclass', 'Title_1', 'Title_2', 

                  'Title_3', 'Title_4', 'Title_5', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 

                  'Age_5', 'Embarked_1', 'Embarked_2', 'FareBin', 'Pclass_x_Sex', 'Pclass_x_AgeBin', 

                  'Is_English_native']



# For KNN (which is very prone to overfitting and not admissible for automatic feature selection)

features_KNN = ['Sex', 'AgeBin', 'Pclass', 'FareBin']



# XGB in the current version is no longer admissible for Sklearn RFECV. I optimized this selection manually.

features_XGB = ['Sex', 'Title', 'Pclass', 'AgeBin', 'Embarked']
# Recursive Feature Elimination with Cross-Validation (RFECV)

# Admissible models

RFECV_models = ['LogReg', 'SVM', 'DecisionTree', 'RandForest']



# Store the RFECV results for each model

optimal_features = {}





for model in RFECV_models:

    features = features_cat if model in cat_models else features_dummy

    X_train = train_data[features]

    y_train = train_data['Survived']

    

    # Create and fit selector that will pick the optimal set of features

    selector = RFECV(estimator=eval(model), scoring=SCORING_METHOD)

    selector.fit(X_train, y_train)

    

    support = selector.support_ # List of Booleans: Is feature selected?

    ranks = selector.ranking_[support] # List of feature importance ranks for selected features

    features = np.array(features)[support] # Unordered list of selected features 

    

    optimal_features[model] = {

        'n':                 selector.n_features_, # number of selected features

        'ranking_selected':  [feature for _,feature in sorted(zip(ranks,features))] # List of selected features ordered by importance

    }

    

# Separate handling of the models that are inadmissible

# or have implemented feature selection:



optimal_features['KNN'] = {

        'n':                 len(features_KNN),

        'ranking_selected':  features_KNN

    }



optimal_features['XGB'] = {

        'n':                 len(features_cat),

        'ranking_selected':  features_XGB

    }

# Define the search space for Bayesian Optimization



space = {}



space['LogReg'] = {

    'penalty':            hp.choice('penalty', ['l1', 'l2']), # Regularization method to prevent overfitting; L1 may eliminate parameters, L2 never does

    'C':                  hp.loguniform('C', np.log(0.001), np.log(100)), # Regularization parameter

    'solver' :            hp.choice('solver', ['liblinear'])

             }



space['SVM'] = {

    'C':                  hp.loguniform('C', np.log(0.001), np.log(10)), # Regularization parameter (using L2 penalty)

    'kernel':             hp.choice('kernel', ['linear'])

             }



space['DecisionTree'] = { 

    'max_features':       hp.choice('max_features', ['log2', 'sqrt', 'auto']), # How the max number of features the alg. considers per split is determined 

    'criterion':          hp.choice('criterion', ['entropy', 'gini']), # How the quality of a split is determined (i.e., measure of impurity)

    'max_depth':          hp.choice('max_depth', np.arange(4, 26, dtype=int)), 

    'min_samples_split':  hp.choice('min_samples_split', np.arange(2, 30, dtype=int)), # Higher values prevent overfitting

    'random_state':       hp.choice('random_state', [RANDOM_STATE])

             }



space['RandForest'] = {

    'max_features':       hp.choice('max_features', ['log2', 'sqrt', 'auto']), # How the max number of features the alg. considers per split is determined

    'criterion':          hp.choice('criterion', ['entropy', 'gini']), # How the quality of a split is determined (i.e., measure of impurity used)

    'max_depth':          hp.choice('max_depth', np.arange(2, 17, dtype=int)), 

    'min_samples_split':  hp.choice('min_samples_split', np.arange(2, 16, dtype=int)), # Higher values prevent overfitting

    'n_estimators':       hp.choice('n_estimators', np.arange(200, 2001, 100, dtype=int)), # Number of trees

    'bootstrap':          hp.choice('bootstrap', [True]),

    'random_state':       hp.choice('random_state', [RANDOM_STATE])

             }



space['XGB'] = {

    'eta':                hp.uniform('eta', 0.05, 0.5), # How strongly trees should be modified based on previous misclassifications

    'max_depth':          hp.choice('max_depth', np.arange(2, 11, dtype=int)), 

    'min_child_weight':   hp.choice('min_child_weight', np.arange(1, 7, dtype=int)), # Minimal weight sum at each internal node

    'gamma':              hp.uniform('gamma', 0, 0.8), # Minimum loss reduction required to make a split

    'colsample_bytree':   hp.uniform('colsample_bytree', 0.2, 0.9), # Fraction of columns to be randomly sampled for each tree

    'colsample_bylevel':  hp.uniform('colsample_bylevel', 0.2, 0.9),

    'colsample_bynode':   hp.uniform('colsample_bynode', 0.2, 0.9),

    'subsample':          hp.uniform('subsample', 0.01, 1), # low values prevent overfitting

    'scale_pos_weight':   hp.uniform('scale_pos_weight', 0.01, 3), # control balance of positive and negative weights

    'seed':               hp.choice('seed', [RANDOM_STATE])

             }



space['KNN'] = {

    'n_neighbors':        hp.choice('n_neighbors', np.arange(1, 31, dtype=int)), # number of neighbors to consider

    'weights':            hp.choice('weights', ['uniform', 'distance']), # possible distance weights for chosen neighbors

    'leaf_size':          hp.choice('leaf_size', np.arange(1, 51, dtype=int))

}
def objective(params):

    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.

       Evaluates loss for a sample of hyperparameters (params) based on cross validation.

    """

    

    # Select training data based on feature optimization (see above)

    X_train = train_data[optimal_features[model]['ranking_selected']]

    y_train = train_data['Survived']

    

    # Perform 5-fold cross validation with AUC-ROC

    classifier = eval(models[model] + '(**params)')

    cv_avg_score = cross_val_score(classifier, X_train, y_train, cv=CV, scoring=SCORING_METHOD).mean()

    

    # Loss is minimized by convention

    loss = 1 - cv_avg_score

    

    # Dictionary with information for evaluation

    return {'loss': loss, 'params': params, 'status': STATUS_OK}





# Optimization Algorithm

tpe_algorithm = tpe.suggest



# Store the Bayesian Optimization results for each model

optimal_hyperparams = {}



for model in models:

    print(model)

    

    # Define number of evaluations

    # RandForest takes by far the longest time, so one should be careful

    MAX_EVALS = 3 if model == 'RandForest' else 50

    

    # Keep track of results

    bayes_trials = Trials()



    # Run optimization

    best = fmin(fn=objective, space=space[model], algo=tpe.suggest, 

                max_evals = MAX_EVALS, trials=bayes_trials, rstate = np.random.RandomState(RANDOM_STATE))

    

    best_params = sorted(bayes_trials.results, key = lambda x: x['loss'])[0]['params']

    

    optimal_hyperparams[model] = {

        'best_params':    best_params,

        'best_estimator': eval(models[model] + '(**best_params)')

    }
# Recursive Feature Elimination with Cross-Validation (RFECV)



for model in RFECV_models:

    features = features_cat if model in cat_models else features_dummy

    X_train = train_data[features]

    y_train = train_data['Survived']

    

    # Create and fit selector that will pick the optimal set of features for the optimal estimator

    estimator = optimal_hyperparams[model]['best_estimator']

    selector = RFECV(estimator=estimator, scoring=SCORING_METHOD)

    selector.fit(X_train, y_train)

    

    support = selector.support_ # List of Booleans: Is feature selected?

    ranks = selector.ranking_[support] # List of feature importance ranks for selected features

    features = np.array(features)[support] # Unordered list of selected features 

    

    optimal_features[model] = {

        'n':                 selector.n_features_, # number of selected features

        'ranking_selected':  [feature for _,feature in sorted(zip(ranks,features))] # List of selected features ordered by importance

    }
def add_performance_score(performance, model_name, classifier, X_train, y_train, cv=CV, scoring=SCORING_METHOD):

    ''' Calculates cross-validation mean performance scores plus standard deviation for classifier with name model_name. 

        Returns an updated performance dictionary.

        Number of validations in k-fold cross val is given by cv.

        Performance metric is specified by scoring.

    '''

    # Get cross-validation mean performance scores

    cv_scores = cross_validate(classifier, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

    

    # Update performance dictionary with: 

    # - Average In-Sample and Out-of-Sample prediction performance

    # - The respective standard deviations

    performance = performance.append({

        'Model':              model_name,

        'In-Sample':          cv_scores['train_score'].mean(),

        'In-Sample_std':      cv_scores['train_score'].std(),

        'Out-of-Sample':      cv_scores['test_score'].mean(),

        'Out-of-Sample_std':  cv_scores['test_score'].std()

    }, ignore_index=True)

    return performance



# Store prediction scores

performance_summary = pd.DataFrame(columns=['Model', 'In-Sample', 'In-Sample_std', 'Out-of-Sample', 'Out-of-Sample_std'])



# Store predictions for Kaggle submission

predict_test = pd.DataFrame()



# Assign train labels

y_train = train_data['Survived']



for model in models:

    # Pick classifier with optimal set of hyperparameters

    classifier = optimal_hyperparams[model]['best_estimator']

    

    # Select train and test data based on optimal set of features

    X_train = train_data[optimal_features[model]['ranking_selected']]

    X_test = test_data[optimal_features[model]['ranking_selected']]

    

    # Get cross-validation mean performance scores plus standard deviation

    # Used for model comparison and selection

    performance_summary = add_performance_score(performance_summary, model, classifier, X_train, y_train)

    

    # Fit model on whole training data set --> predictions will be used for Kaggle submission

    classifier.fit(X_train, y_train)

    predict_test[model] = classifier.predict(X_test)



# Benchmark (dummy) model 1: proportional

Proportional = DummyClassifier(strategy='stratified')

performance_summary = add_performance_score(performance_summary, model_name='Proportional', 

                                            classifier=Proportional, X_train=train_data, y_train=y_train)



# Benchmark (dummy) model 1: all_dead

All_Dead = DummyClassifier(strategy='most_frequent')

performance_summary = add_performance_score(performance_summary, model_name='All_Dead', 

                                            classifier=All_Dead, X_train=train_data, y_train=y_train)



# Get an ordered table summarizing the model performances    

performance_summary.sort_values(by='Out-of-Sample', ascending=False)
# Best individual model

model = performance_summary.loc[performance_summary['Out-of-Sample'].idxmax(), 'Model']

predictions = predict_test[model]

submission_name = 'submission_' + model + '.csv'



submission = pd.DataFrame({

    'PassengerId': test_data['PassengerId'],

    'Survived': predictions

})

submission.to_csv(submission_name, index=False)