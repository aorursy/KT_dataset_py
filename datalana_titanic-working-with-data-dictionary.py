import numpy as np

import pandas as pd

import sklearn 



import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import platform

print('Versions:')

print('  python', platform.python_version())

n = ('numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn')

nn = (np, pd, sklearn, mpl, sns)

for a, b in zip(n, nn):

    print('  --', str(a), b.__version__)
#pandas styling

pd.set_option('colheader_justify', 'left')

pd.set_option('precision', 0)

pd.options.display.float_format = '{:,.2f}'.format

pd.set_option('display.max_colwidth', -1)
#pd.reset_option('all')
#seaborn syling

sns.set_style('whitegrid', { 'axes.axisbelow': True, 'axes.edgecolor': 'black', 'axes.facecolor': 'white',

        'axes.grid': True, 'axes.labelcolor': 'black', 'axes.spines.bottom': True, 'axes.spines.left': True,

        'axes.spines.right': False, 'axes.spines.top': False, 'figure.facecolor': 'white', 

        #'font.family': ['sans-serif'], 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

        'grid.color': 'grey', 'grid.linestyle': ':', 'image.cmap': 'rocket', 'lines.solid_capstyle': 'round',

        'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'text.color': 'black', 

        'xtick.top': False, 'xtick.bottom': True, 'xtick.color': 'navy', 'xtick.direction': 'out', 

        'ytick.right': False,    'ytick.left': True, 'ytick.color': 'navy', 'ytick.direction': 'out'})
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process 

from sklearn import feature_selection, model_selection, metrics

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from xgboost import XGBClassifier
train_raw = pd.read_csv('../input/titanic/train.csv')

test_raw = pd.read_csv('../input/titanic/test.csv')

len(train_raw) + len(test_raw)
df = pd.concat(objs=[train_raw, test_raw], axis=0)

df.shape
#ddict = pd.read_csv('../input/Titanic_Data_Dictionary_ready.csv', index_col=0)

#ddict
col1 = test_raw['PassengerId'] # will need for a submission

df.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

df['Embarked'] = df['Embarked'].fillna('S')

df.isnull().sum().to_frame().T
#family sizes

df['Fsize'] =  df['Parch'] + df['SibSp'] + 1
#titles

df['Surname'], df['Name'] = zip(*df['Name'].apply(lambda x: x.split(',')))

df['Title'], df['Name'] = zip(*df['Name'].apply(lambda x: x.split('.')))



titles = (df['Title'].value_counts() < 10)

df['Title'] = df['Title'].apply(lambda x: ' Misc' if titles.loc[x] == True else x)

df['Title'].value_counts().to_frame().T
# ticket set (how many person in one ticket)

# if one ticket has family members only, it's "monotinic"; if not - "mixed"

df['Tname'] = df['Ticket']

df['Tset']=0



for t in df['Tname'].unique():

    if df['Surname'].loc[(df['Tname']==t)].nunique() != 1:

        df['Tset'].loc[(df['Tname']==t)] = 'mixed'

    else: 

        df['Tset'].loc[(df['Tname']==t)] = 'monotonic'



for t in df['Tname'].unique():

    if df['Surname'].loc[(df['Tname']==t)].nunique() != 1:

        df['Tset'].loc[(df['Tname']==t)] = 'mixed'

    else: 

        df['Tset'].loc[(df['Tname']==t)] = 'monotonic'
#price and 

for t in df['Ticket'].unique():

    df['Ticket'].loc[(df['Ticket']==t)] = len(df.loc[(df['Ticket']==t)]) 

df['Price'] = df['Fare'] / df['Ticket']

#renaming "Ticket"

df.rename(columns={'Ticket':'Tgroup'}, inplace=True)
#deleting useless again

df.drop(['Parch', 'SibSp', 'Name', 'Surname', 'Tname', 'Fare'], axis=1, inplace=True)
df.head(2)
df.dtypes.to_frame().sort_values([0]).T
#code categorical data

label = LabelEncoder()

cols = df.dtypes[df.dtypes == 'object'].index.tolist()

for col in cols:

    df[col] = label.fit_transform(df[col])
#binning

df['Price'] = pd.qcut(df['Price'], 4)

df['Age'] = pd.cut(df['Age'].astype(int), 5)
#code binning data

df['Age'] = label.fit_transform(df['Age'])

df['Price'] = label.fit_transform(df['Price'])
df.head()
a = len(train_raw)

train = df[:a]

test = df[a:]
train_raw.shape[0] == train.shape[0]
test.drop(['Survived'], axis=1, inplace=True)

test_raw.shape[0] == test.shape[0]
X = train.drop(['Survived'], axis=1).columns.to_list()

y = ['Survived']
#Machine Learning Algorithm initialization

MLA = [ #Ensemble Methods

        ensemble.AdaBoostClassifier(), ensemble.BaggingClassifier(), ensemble.ExtraTreesClassifier(),

        ensemble.GradientBoostingClassifier(), ensemble.RandomForestClassifier(),

        #Gaussian Processes

        gaussian_process.GaussianProcessClassifier(),

        #GLM

        linear_model.LogisticRegressionCV(), linear_model.PassiveAggressiveClassifier(),

        linear_model.RidgeClassifierCV(), linear_model.SGDClassifier(), linear_model.Perceptron(),

        #Navies Bayes

        naive_bayes.BernoulliNB(), naive_bayes.GaussianNB(),

        #Nearest Neighbor

        neighbors.KNeighborsClassifier(),

        #SVM

        svm.SVC(probability=True), svm.NuSVC(probability=True), svm.LinearSVC(),

        #Trees    

        tree.DecisionTreeClassifier(), tree.ExtraTreeClassifier(),

        #Discriminant Analysis

        discriminant_analysis.LinearDiscriminantAnalysis(), discriminant_analysis.QuadraticDiscriminantAnalysis(),

        #xgboost

        XGBClassifier() ]
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)



mla = pd.DataFrame(columns=['Name','TestScore','ScoreTime','FitTime','Parameters'])

prediction = train[y]



i = 0

for alg in MLA:

    name = alg.__class__.__name__

    mla.loc[i, 'Name'] = name

    mla.loc[i, 'Parameters'] = str(alg.get_params())

    

    cv_results = model_selection.cross_validate(alg, train[X], train[y], cv=cv_split)

        

    mla.loc[i, 'FitTime'] = cv_results['fit_time'].mean()

    mla.loc[i, 'ScoreTime'] = cv_results['score_time'].mean()

    mla.loc[i, 'TestScore'] = cv_results['test_score'].mean()



    alg.fit(train[X], train[y])

    prediction[name] = alg.predict(train[X])    

    i += 1



mla = mla.sort_values('TestScore', ascending=False).reset_index(drop=True)

mla
# assing parameters

param_grid = {'criterion': ['gini', 'entropy'],  #default is gini

              #'splitter': ['best', 'random'], #default is best

              'max_depth': [2,4,6,8,10,None], #default is none

              #'min_samples_split': [2,5,10,.03,.05], #default is 2

              #'min_samples_leaf': [1,5,10,.03,.05], #default is 1

              #'max_features': [None, 'auto'], #default none or all

              'random_state': [0]}



#choose best model with grid_search

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split)

tune_model.fit(train[X], train[y])

print('Parameters: ', tune_model.best_params_)
clf = tree.DecisionTreeClassifier()

results = model_selection.cross_validate(clf, train[X], train[y], cv=cv_split)

clf.fit(train[X], train[y])

results['test_score'].mean()*100



#feature selection

fs = feature_selection.RFECV(clf, step=1, scoring='accuracy', cv=cv_split)

fs.fit(train[X], train[y])



#transform x and y to fit a new model

X = train[X].columns.values[fs.get_support()]

results = model_selection.cross_validate(clf, train[X], train[y], cv=cv_split)



print('Shape New: ', train[X].shape) 

print('Features to use: ', X)
#tune parameters

tuned = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv=cv_split)

tuned.fit(train[X], train[y])



param_grid = tuned.best_params_

param_grid
clf = ensemble.GradientBoostingClassifier()

results = model_selection.cross_validate(clf, train[X], train[y], cv=cv_split)

clf.fit(train[X], train[y])

results['test_score'].mean()*100
test['Survived'] = clf.predict(test[X])
submit = pd.DataFrame({ 'PassengerId' : col1, 'Survived': test['Survived'] }).set_index('PassengerId')

submit['Survived'] = submit['Survived'].astype('int')

submit.to_csv('submission.csv')