# NumPy is the fundamental package for scientific computing
import numpy as np
# Pandas is a high-level data manipulation tool
import pandas as pd
# Matplotlib is a plotting library
import matplotlib.pyplot as plt
# Seaborn is a Python data visualization library based on matplotlib
import seaborn as sns

# import Warnings library and disable all warnings
import warnings 
warnings.filterwarnings('ignore')

# import Os and display the list of available data
import os
print(os.listdir('../input'))
# load train and test data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# show train and test shape
print(f'Train data shape: {train_data.shape}\nTest data shape: {test_data.shape}')
# show the first five lines of train data
train_data.head()
# show the first five lines of test data
test_data.head()
# separate the PassengerId feature, which is needed to save the final result
test_id = test_data['PassengerId']
# combine train and test data for further analysis and engineering
dataset = pd.concat((train_data, test_data), sort=True)
# drop PassengerId feature
dataset.drop(['PassengerId'], axis=1, inplace=True)
# show information about our dataset
dataset.info()
# describe our dataset
dataset.describe()
# correlation matrix between numerical values and Survived
fig, ax = plt.subplots(figsize=(10, 8))
plot = sns.heatmap(train_data[['Survived','SibSp','Parch','Age','Fare']].corr(), annot=True, cmap='BuPu', ax=ax)
# explore Age distribution
plot = sns.kdeplot(train_data['Age'][(train_data['Survived'] == 0) &
                   (train_data['Age'].notnull())], color='darkorchid', shade=True)
plot = sns.kdeplot(train_data['Age'][(train_data['Survived'] == 1) & 
                   (train_data['Age'].notnull())], ax=plot, color='darkblue', shade=True)
plot.set_xlabel('Age')
plot.set_ylabel('Frequency')
plot = plot.legend(['Not Survived','Survived'])
# fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# get indexes of null
null_age = list(dataset['Age'][dataset['Age'].isnull()].index)

for i in null_age:
    # get median value of age
    age_median = dataset['Age'].median()
    # get median age of similar rows according to Pclass, Parch and SibSp
    age_predict = dataset['Age'][((dataset['SibSp'] == dataset.iloc[i]['SibSp']) &
                                  (dataset['Parch'] == dataset.iloc[i]['Parch']) & 
                                  (dataset['Pclass'] == dataset.iloc[i]['Pclass']))].median()
    # if exists a similar value then fill the value of age_predict
    if not np.isnan(age_predict):
        dataset.loc[i, 'Age'] = age_predict
    # if not exists fill the value of age_median
    else:
        dataset.loc[i, 'Age'] = age_median
# divide Age feature
dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
# explore Age groups
plot = sns.catplot(x='Age', y='Survived', data=dataset, kind='bar', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# convert to indicator variables
dataset = pd.get_dummies(dataset, columns=['Age'])
# explore SibSp vs Survival
plot = sns.catplot(x='SibSp', y='Survived', data=dataset, kind='bar', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# explore Parch vs Survival
plot = sns.catplot(x='Parch', y='Survived', data=dataset, kind='bar', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# create Fsize feature
dataset['Fsize'] = dataset['SibSp'] + dataset['Parch'] + 1
# explore Fsize vs Survived
plot = sns.catplot(x='Fsize', y='Survived', data=dataset, kind='bar', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# divide Fsize feature
dataset.loc[dataset['Fsize'] == 1, 'Fsize'] = 0
dataset.loc[dataset['Fsize'] == 2, 'Fsize'] = 1
dataset.loc[(dataset['Fsize'] >=3) & (dataset['Fsize'] <=4), 'Fsize'] = 2
dataset.loc[dataset['Fsize'] >= 5, 'Fsize'] = 3
# explore Fsize groups vs Survived
plot = sns.catplot(x='Fsize', y='Survived', data=dataset, kind='bar', palette='BuPu')
plot = plot.set_xticklabels(['Single', 'Small', 'Medium', 'Large'])
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# convert to indicator variables
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MediumF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
# fill Fare with the median
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
# explore Fare distribution
plot = sns.distplot(dataset['Fare'], 
                    label='Skewness: %.2f'%(dataset['Fare'].skew()), color='darkblue')
plot = plot.legend(loc='best')
# transform feature with the log function
dataset['Fare'] = dataset['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
# explore Fare distribution again
plot = sns.distplot(dataset['Fare'], 
                    label='Skewness: %.2f'%(dataset['Fare'].skew()), color='darkblue')
plot = plot.legend(loc='best')
# explore Pclass vs Survived
plot = sns.catplot(x='Pclass',y='Survived', 
                   data=train_data, kind='bar', palette='BuPu')
plot.despine(left=True)
plot.set_ylabels('Survival Rate')

# explore Pclass vs Survived by Sex
plot = sns.catplot(x='Sex', y='Survived', hue='Pclass', 
                   data=train_data, kind='bar', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# convert to indicator variables
dataset = pd.get_dummies(dataset, columns = ['Pclass'])
# explore Embarked frequency histogram
plot = sns.catplot('Embarked', data=dataset, kind='count', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Count')
# fill Embarked with the most frequent value
dataset['Embarked'] = dataset['Embarked'].fillna('S')
# explore Embarked vs Survived
plot = sns.catplot(x='Embarked', y='Survived', data=train_data, kind='bar', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# explore Age vs Embarked
plot = sns.catplot('Sex', col='Embarked', data=train_data, kind='count', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Count')

# explore Pclass vs Embarked
plot = sns.catplot('Pclass', col='Embarked', data=train_data, kind='count', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Count')
# convert to indicator variables
dataset = pd.get_dummies(dataset, columns = ['Embarked'])
# explore Sex vs Survived
plot = sns.catplot(x='Sex',y='Survived', data=train_data, kind='bar', palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# convert to indicator variables
dataset = pd.get_dummies(dataset, columns = ['Sex'])
# number of null values
dataset['Cabin'].isnull().sum()
# show the first five not null values
dataset['Cabin'][dataset['Cabin'].notnull()].head()
# replace the Cabin number by the type of cabin 'X' if not
dataset['Cabin'] = pd.Series([x[0] if not pd.isnull(x) else 'X' for x in dataset['Cabin']])
# explore Cabin frequency histogram
plot = sns.countplot(dataset['Cabin'], 
                     order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'], palette='BuPu')
# explore Cabin vs Survived
plot = sns.catplot(x='Cabin', y='Survived', data=dataset, kind='bar', 
                   order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'], palette='BuPu')
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# convert to indicator variables
dataset = pd.get_dummies(dataset, columns=['Cabin'], prefix='Cabin')
# show the first five values
dataset['Name'].head()
# create Title feature from Name
dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# explore Title frequency histogram
fig, ax = plt.subplots(figsize=(10, 6))
plot = sns.countplot(x='Title',data=dataset, palette='BuPu', 
                     order=dataset['Title'].value_counts(ascending=True).index, ax=ax)
plot = plt.setp(plot.get_xticklabels(), rotation=90) 
# convert Title feature to categorical values
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt',
                                             'Col','Don', 'Dr', 'Major', 'Rev', 'Sir',
                                             'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].map({'Mr': 0, 'Miss': 1, 'Ms': 1, 'Mlle': 1,
                                         'Mrs': 1, 'Mme': 1, 'Master': 2, 'Rare': 3})
# explore Title frequency histogram
plot = sns.countplot(dataset['Title'], palette='BuPu')
plot = plot.set_xticklabels(['Mr', 'Miss', 'Master', 'Rare'])
# explore Title vs Survived
plot = sns.catplot(x='Title', y='Survived', data=dataset, kind='bar', palette='BuPu')
plot = plot.set_xticklabels(['Mr', 'Miss', 'Master', 'Rare'])
plot.despine(left=True)
plot = plot.set_ylabels('Survival Rate')
# drop Name feature
dataset.drop(labels=['Name'], axis=1, inplace=True)
# show the first five values
dataset['Ticket'].head()
# treat Ticket by extracting the ticket prefix. 'X' if not prefix
tickets = []
for i in list(dataset['Ticket']):
    if not i.isdigit() :
        tickets.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        tickets.append("X")
        
dataset['Ticket'] = tickets
# explore Ticket frequency histogram
fig, ax = plt.subplots(figsize=(12, 6))
plot = sns.countplot(x='Ticket', data=dataset, 
                     order=dataset['Ticket'].value_counts(ascending=True).index, palette='BuPu')
plot = plt.setp(plot.get_xticklabels(), rotation=90)
# convert to indicator variables
dataset = pd.get_dummies(dataset, columns=['Ticket'])
# show the first five lines of dataset
dataset.head()
# separate our dataset on train and test data
train = dataset[:train_data.shape[0]]
test = dataset[train_data.shape[0]:]
test.drop('Survived', axis=1, inplace=True)
# separate train features and label
train['Survived'] = train['Survived'].astype(int)

Y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
# initialization algorithms
algorithms = {# Ensemble Methods
              ensemble.AdaBoostClassifier(),
              ensemble.BaggingClassifier(),
              ensemble.ExtraTreesClassifier(),
              ensemble.GradientBoostingClassifier(),
              ensemble.RandomForestClassifier(),

              # Gaussian Processes
              gaussian_process.GaussianProcessClassifier(),
    
              # Generalized Linear Models
              linear_model.LogisticRegressionCV(),
              linear_model.PassiveAggressiveClassifier(),
              linear_model.RidgeClassifierCV(),
              linear_model.SGDClassifier(),
              linear_model.Perceptron(),
    
              # Navies Bayes
              naive_bayes.BernoulliNB(),
              naive_bayes.GaussianNB(),
    
              # Nearest Neighbor
              neighbors.KNeighborsClassifier(),
    
              # Support Vector Machine
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
              XGBClassifier() }
# split dataset in cross-validation
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, 
                                        train_size=.6, random_state=0)
# create table to compare algorithms
algorithm_results = pd.DataFrame(columns=['Name', 'Parameters', 
                                          'Train Accuracy', 'Test Accuracy', 'STD'])
# index through algorithms and save performance to table
index = 0

predictions = pd.DataFrame()
predictions['Target'] = Y_train

for alg in algorithms:

    # set name and parameters
    name = alg.__class__.__name__
    algorithm_results.loc[index, 'Name'] = name
    algorithm_results.loc[index, 'Parameters'] = str(alg.get_params())

    # score model with cross validation
    cv_results = model_selection.cross_validate(alg, X_train, Y_train, cv=cv_split)

    algorithm_results.loc[index, 'Train Accuracy'] = cv_results['train_score'].mean()
    algorithm_results.loc[index, 'Test Accuracy'] = cv_results['test_score'].mean()   

    algorithm_results.loc[index, 'STD'] = cv_results['test_score'].std()

    alg.fit(X_train, Y_train)
    
    # algoritm prediction
    predictions[name] = alg.predict(X_train)
    
    index+=1

    
# sort table
algorithm_results.sort_values(by=['Test Accuracy'], ascending=False, inplace=True)
# print algorithms results table
algorithm_results
# explore algorithms accuracy score
fig, ax = plt.subplots(figsize=(14, 8))

plot = sns.barplot(x='Test Accuracy', y='Name', data=algorithm_results, palette='BuPu', 
                   orient='h', **{'xerr': algorithm_results['STD']})

plt.title('Algorithms Accuracy Score')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
# correlation matrix between predictions and Survived
fig, ax = plt.subplots(figsize=(16, 15))
plot = sns.heatmap(predictions.corr(), annot=True, cmap='BuPu', ax=ax)
# AdaBoostClassifier grid search
#ada = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(), random_state=0)

#ada_params = {'base_estimator__criterion': ['gini', 'entropy'],
#              'base_estimator__splitter':   ['best', 'random'],
#              'n_estimators': [1, 2, 5, 10, 20, 50, 100, 300],
#              'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.25, 0.5],
#              'algorithm': ['SAMME', 'SAMME.R']}

#ada_grid = model_selection.GridSearchCV(ada, param_grid=ada_params, cv=cv_split,
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#ada_grid.fit(X_train, Y_train)

#ada_best = ada_grid.best_estimator_

#print(f'The best parameter for AdaBoostClassifier is {ada_grid.best_params_}')
#print(f'The best score for AdaBoostClassifier is {ada_grid.best_score_}')
# AdaBoostClassifier
ada_best = ensemble.AdaBoostClassifier(
    tree.DecisionTreeClassifier(criterion='entropy', splitter='best'), 
    algorithm='SAMME.R', learning_rate=0.1, n_estimators=100, random_state=0)

ada_best.fit(X_train, Y_train)
score = model_selection.cross_validate(ada_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for AdaBoostClassifier is {score}')
# BaggingClassifier grid search
#bc = ensemble.BaggingClassifier(random_state=0)

#bc_params = {'max_samples': [0.1, 0.25, 0.5, 0.75, 1.0],
#             'max_features': [0.1, 0.25, 0.5, 0.75, 1.0],
#             'n_estimators': [1, 2, 5, 10, 20, 50, 100, 300]}

#bc_grid = model_selection.GridSearchCV(bc, param_grid=bc_params, cv=cv_split, 
#                                       scoring='accuracy', n_jobs=8, verbose=1)
#bc_grid.fit(X_train, Y_train)

#bc_best = bc_grid.best_estimator_

#print(f'The best parameter for BaggingClassifier is {bc_grid.best_params_}')
#print(f'The best score for BaggingClassifier is {bc_grid.best_score_}')
# BaggingClassifier
bc_best = ensemble.BaggingClassifier(random_state=0, max_samples=0.5, 
                                     max_features=1.0, n_estimators=50)

bc_best.fit(X_train, Y_train)
score = model_selection.cross_validate(bc_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for BaggingClassifier is {score}')
# ExtraTreesClassifier grid search
#etc = ensemble.ExtraTreesClassifier(random_state=0)

#etc_params = {'max_depth': [None, 2, 4, 6, 8, 10],
#              'max_features': [0.1, 0.25, 0.5, 0.75, 1.0],
#              'n_estimators': [1, 2, 5, 10, 20, 50, 100, 300],
#              'criterion': ['gini', 'entropy']}

#etc_grid = model_selection.GridSearchCV(etc, param_grid=etc_params, cv=cv_split, 
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#etc_grid.fit(X_train, Y_train)

#etc_best = etc_grid.best_estimator_

#print(f'The best parameter for ExtraTreesClassifier is {etc_grid.best_params_}')
#print(f'The best score for ExtraTreesClassifier is {etc_grid.best_score_}')
# ExtraTreesClassifier
etc_best = ensemble.ExtraTreesClassifier(random_state=0, max_depth=6, max_features=0.25,
                                         n_estimators=50, criterion='entropy')

etc_best.fit(X_train, Y_train)
score = model_selection.cross_validate(etc_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for ExtraTreesClassifier is {score}')
# GradientBoostingClassifier grid search
#gbc = ensemble.GradientBoostingClassifier(random_state=0)

#gbc_params = {'max_depth': [None, 2, 4, 6, 8, 10],
#              'max_features': [0.1, 0.25, 0.5, 0.75, 1.0],
#              'learning_rate': [0.01, 0.025, 0.05, 0.1],
#              'n_estimators': [1, 2, 5, 10, 20, 50, 100, 300]}

#gbc_grid = model_selection.GridSearchCV(gbc, param_grid=gbc_params, cv=cv_split, 
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#gbc_grid.fit(X_train, Y_train)

#gbc_best = gbc_grid.best_estimator_

#print(f'The best parameter for GradientBoostingClassifier is {gbc_grid.best_params_}')
#print(f'The best score for GradientBoostingClassifier is {gbc_grid.best_score_}')
# GradientBoostingClassifier
gbc_best = ensemble.GradientBoostingClassifier(random_state=0, max_depth=4, max_features=0.75,
                                               n_estimators=100, learning_rate=0.05)

gbc_best.fit(X_train, Y_train)
score = model_selection.cross_validate(gbc_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for GradientBoostingClassifier is {score}')
# RandomForestClassifier grid search
#rfc = ensemble.RandomForestClassifier(random_state=0)

#rfc_params = {'max_depth': [None, 2, 4, 6, 8, 10],
#              'max_features': [0.1, 0.5, 1.0],
#              'min_samples_split': [2, 3, 10],
#              'min_samples_leaf': [1, 3, 10],
#              'criterion': ['gini', 'entropy'],
#              'n_estimators': [1, 2, 5, 10, 20, 50, 100, 300]}

#rfc_grid = model_selection.GridSearchCV(rfc, param_grid=rfc_params, cv=cv_split, 
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#rfc_grid.fit(X_train, Y_train)

#rfc_best = rfc_grid.best_estimator_

#print(f'The best parameter for RandomForestClassifier is {rfc_grid.best_params_}')
#print(f'The best score for RandomForestClassifier is {rfc_grid.best_score_}')
# RandomForestClassifier
rfc_best = ensemble.RandomForestClassifier(random_state=0, max_depth=8, max_features=1.0,
                                           min_samples_split=10, min_samples_leaf=1,
                                           criterion='entropy', n_estimators=10)

rfc_best.fit(X_train, Y_train)
score = model_selection.cross_validate(rfc_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for RandomForestClassifier is {score}')
# GaussianProcessClassifier grid search
#gpc = gaussian_process.GaussianProcessClassifier(random_state=0)
    
#gpc_params = {'max_iter_predict': [1, 2, 5, 10, 20, 50, 100, 300],
#              'n_restarts_optimizer': [1, 2, 3, 4, 5, 10]}

#gpc_grid = model_selection.GridSearchCV(gpc, param_grid=gpc_params, cv=cv_split, 
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#gpc_grid.fit(X_train, Y_train)

#gpc_best = gpc_grid.best_estimator_

#print(f'The best parameter for GaussianProcessClassifier is {gpc_grid.best_params_}')
#print(f'The best score for GaussianProcessClassifier is {gpc_grid.best_score_}')
# GaussianProcessClassifier
gpc_best = gaussian_process.GaussianProcessClassifier(random_state=0, max_iter_predict=5,
                                                      n_restarts_optimizer=1)

gpc_best.fit(X_train, Y_train)
score = model_selection.cross_validate(gpc_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for GaussianProcessClassifier is {score}')
# LogisticRegressionCV grid search
#lr = linear_model.LogisticRegressionCV(random_state=0)
    
#lr_params = {'fit_intercept': [True, False],
#             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

#lr_grid = model_selection.GridSearchCV(lr, param_grid=lr_params, cv=cv_split,
#                                       scoring='accuracy', n_jobs=8, verbose=1)
#lr_grid.fit(X_train, Y_train)

#lr_best = lr_grid.best_estimator_

#print(f'The best parameter for LogisticRegressionCV is {lr_grid.best_params_}')
#print(f'The best score for LogisticRegressionCV is {lr_grid.best_score_}')
# LogisticRegressionCV
lr_best = linear_model.LogisticRegressionCV(random_state=0, fit_intercept=True,
                                            solver='liblinear')

lr_best.fit(X_train, Y_train)
score = model_selection.cross_validate(lr_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for LogisticRegressionCV is {score}')
# BernoulliNB grid search
#bnb = naive_bayes.BernoulliNB()
    
#bnb_params = {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}

#bnb_grid = model_selection.GridSearchCV(bnb, param_grid=bnb_params, cv=cv_split,
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#bnb_grid.fit(X_train, Y_train)

#bnb_best = bnb_grid.best_estimator_

#print(f'The best parameter for BernoulliNB is {bnb_grid.best_params_}')
#print(f'The best score for BernoulliNB is {bnb_grid.best_score_}')
# BernoulliNB
bnb_best = naive_bayes.BernoulliNB(alpha=0.25)

bnb_best.fit(X_train, Y_train)
score = model_selection.cross_validate(bnb_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for BernoulliNB is {score}')
# KNeighborsClassifier grid search
#knn = neighbors.KNeighborsClassifier()
    
#knn_params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#              'weights': ['uniform', 'distance'],
#              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

#knn_grid = model_selection.GridSearchCV(knn, param_grid=knn_params, cv=cv_split,
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#knn_grid.fit(X_train, Y_train)

#knn_best = knn_grid.best_estimator_

#print(f'The best parameter for KNeighborsClassifier is {knn_grid.best_params_}')
#print(f'The best score for KNeighborsClassifier is {knn_grid.best_score_}')
# KNeighborsClassifier
knn_best = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='brute')

knn_best.fit(X_train, Y_train)
score = model_selection.cross_validate(knn_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for KNeighborsClassifier is {score}')
# SVC grid search
#svc = svm.SVC(probability=True, random_state=0)
    
#svc_params = {'C': [1, 2, 3, 4, 5],
#              'gamma': [0.1, 0.25, 0.5, 0.75, 1.0],
#              'decision_function_shape': ['ovo', 'ovr'],
#              'probability': [True]}

#svc_grid = model_selection.GridSearchCV(svc, param_grid=svc_params, cv=cv_split, 
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#svc_grid.fit(X_train, Y_train)

#svc_best = svc_grid.best_estimator_

#print(f'The best parameter for SVC is {svc_grid.best_params_}')
#print(f'The best score for SVC is {svc_grid.best_score_}')
# SVC
svc_best = svm.SVC(random_state=0, C=1, gamma=0.1, 
                   decision_function_shape='ovo', probability=True)

svc_best.fit(X_train, Y_train)
score = model_selection.cross_validate(svc_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for SVC is {score}')
# XGBClassifier grid search
#xgb = XGBClassifier(random_state=0)
    
#xgb_params = {'learning_rate': [0.01, 0.025, 0.05, 0.1],
#              'max_depth':  [1, 2, 3, 4, 5, 6, 8, 10],
#              'n_estimators': [1, 2, 5, 10, 20, 50, 100, 300]}

#xgb_grid = model_selection.GridSearchCV(xgb, param_grid=xgb_params, cv=cv_split,
#                                        scoring='accuracy', n_jobs=8, verbose=1)
#xgb_grid.fit(X_train, Y_train)

#xgb_best = xgb_grid.best_estimator_

#print(f'The best parameter for XGBClassifier is {xgb_grid.best_params_}')
#print(f'The best score for XGBClassifier is {xgb_grid.best_score_}')
# XGBClassifier
xgb_best = XGBClassifier(random_state=0, learning_rate=0.025, max_depth=5, n_estimators=300)

xgb_best.fit(X_train, Y_train)
score = model_selection.cross_validate(xgb_best, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for XGBClassifier is {score}')
# combine our models
vote_estimators = {# Ensemble Methods
                   ('ada', ada_best),
                   ('bc', bc_best),
                   ('etc', etc_best),
                   ('gbc', gbc_best),
                   ('rfc', rfc_best),

                   # Gaussian Processes
                   ('gpc', gbc_best),
    
                   # Generalized Linear Models
                   ('lr', lr_best),
    
                   # Navies Bayes
                   ('bnb', bnb_best),
    
                   # Nearest Neighbor
                   ('knn', knn_best),
    
                   # Support Vector Machine
                   ('svc', svc_best),
    
                   # XGBoost
                   ('xgb', xgb_best) }
# fit models with hard voting
hard_estimators = ensemble.VotingClassifier(estimators=vote_estimators, voting='hard')
hard_estimators.fit(X_train, Y_train)

score = model_selection.cross_validate(hard_estimators, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for Hard Estimators is {score}')
# fit models with soft voting
soft_estimators = ensemble.VotingClassifier(estimators=vote_estimators, voting='soft')
soft_estimators.fit(X_train, Y_train)

score = model_selection.cross_validate(soft_estimators, X_train, Y_train, cv=cv_split)['test_score'].mean()

print(f'The best score for Soft Estimators is {score}')
# predict and save results
test_survived = pd.Series(hard_estimators.predict(test), name='Survived')
results = pd.concat([test_id, test_survived], axis=1)

results.to_csv('titanic_result.csv', index=False)